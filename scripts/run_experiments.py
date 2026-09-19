"""Plan, validate, train, resume or evaluate explicitly local TexJEPA experiments.

Default invocation only prints a plan. Execution is opt-in, never downloads
assets, and keeps the bounded synthetic fixture profile separate from real data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset

from configs.experiment import load_experiment_config, normalize_experiment_config, config_digest
from data import (LocalManifestCXRDataset, collate_cxr, deterministic_split,
                  inspect_manifest, load_explicit_split)
from evaluation.lesion_occlusion import evaluate_lesion_occlusion, occlude
from evaluation.mitigation import evaluate_mitigations
from evaluation.robustness import PerturbationSpec, evaluate_robustness
from evaluation.token_diagnostics import compare_token_drift, gradient_activation_saliency
from models.backbones import build_backbone, validate_backbone_spec, BackboneAdapter
from models.checkpoints import checkpoint_sha256
from models.nca import NoiseConsistencyAdapter
from perturbations import gaussian_noise
from probing import ProbeConfig, fit_probe
from probing.trainer import ProbeModel
from report.build_tables import build_tables
from scripts.run_all import configured_perturbations, _jsonable, _versions, _AdaptedEncoder
from training import TrainingConfig, load_training_model, train_nca_resumable
from training.output import resume_checkpoint, restore_latest_pointer


class CompleteBatchSampler(Sampler):
    """Shuffle all training examples while avoiding a singleton BN batch.

    With batch_size=2 and odd N the final batch has 3 examples. Otherwise every
    batch is <=batch_size; no labels or examples disappear via drop_last.
    """
    def __init__(self, size, batch_size, generator):
        if size < 2 or batch_size < 2:
            raise ValueError('Probe training requires at least two samples and batch_size>=2')
        self.size, self.batch_size, self.generator = size, batch_size, generator

    def __iter__(self):
        indices = torch.randperm(self.size, generator=self.generator).tolist()
        chunks = [indices[i:i+self.batch_size] for i in range(0,self.size,self.batch_size)]
        if len(chunks[-1]) == 1:
            if len(chunks[-2]) > 2:
                chunks[-1].insert(0,chunks[-2].pop())
            else:
                chunks[-2].extend(chunks.pop())
        return iter(chunks)

    def __len__(self):
        count=(self.size+self.batch_size-1)//self.batch_size
        return count-1 if self.size % self.batch_size == 1 and self.batch_size == 2 else count


class DeviceBatches:
    def __init__(self, loader, device):
        self.loader, self.device, self.dataset = loader, device, loader.dataset
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            yield {key:value.to(self.device) if key in ('image','labels') else value
                   for key,value in batch.items()}


def _atomic_json(path, value):
    path=Path(path)
    if path.is_symlink():
        raise ValueError('Refusing to write through an output symlink')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(_jsonable(value),stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _asset_paths(config):
    d=config['data']
    assets=[('labeled_manifest',d['manifest']),('image_root',d['image_root'])]
    for key in ('unlabeled_manifest','unlabeled_root'):
        if d[key]: assets.append((key,d[key]))
    if d['split']['path']: assets.append(('split',d['split']['path']))
    for model in config['models']:
        spec=model['backbone']
        if spec.get('checkpoint'): assets.append((model['name']+' checkpoint',spec['checkpoint']['path']))
        for key in ('model_path','repo_path'):
            if spec.get(key): assets.append((model['name']+' '+key,spec[key]))
    post=config['post_training']
    if post['enabled']:
        if post['baseline_checkpoint']: assets.append(('post baseline',post['baseline_checkpoint']))
        assets.extend((key+' parent',path) for key,path in post['parent_checkpoints'].items())
    return assets


def plan_experiment(config):
    """No model construction, image decoding, checkpoint loading or CUDA calls."""
    config=normalize_experiment_config(config)
    jobs=[]
    if config['post_training']['enabled']:
        jobs.extend({'kind':'post_training','variant':v,'epochs':config['post_training']['epochs'][v]}
                    for v in ('v4','v5','v6') if v in config['post_training']['variants'])
    names=[m['name'] for m in config['models']]
    if config['post_training']['enabled']:
        names.extend('TexJEPA-'+v for v in config['post_training']['variants'])
    for name in names:
        for protocol in config['protocols']:
            jobs.append({'kind':'probe_and_robustness','model':name,'protocol':protocol,
                         'epochs':config['probe'][protocol+'_epochs']})
        if 'linear' in config['protocols']:
            for kind,enabled in [('lesion_occlusion',config['lesion']['enabled']),
                                 ('input_smoothing',config['mitigation']['enabled']),
                                 ('train_aug',config['mitigation']['enabled'] and config['mitigation']['train_aug']),
                                 ('nca',config['nca']['enabled']),('token_diagnostics',True)]:
                if enabled: jobs.append({'kind':kind,'model':name})
    return {'mode':'plan','config_sha256':config_digest(config),'execution_started':False,
            'downloads_performed':False,'device':config['runtime']['device'],'jobs':jobs,
            'assets':[{'role':role,'path':path,'exists':Path(path).exists()} for role,path in _asset_paths(config)],
            'evidence_boundary':['Paper numerical results have not been reproduced.',
                'Historical author checkpoints, exact class mapping and original split IDs are external inputs.',
                'Unreported optimizer, normalization, augmentation and NCA settings are recorded implementation choices.']}


def _reject_synthetic(metadata, label):
    from models.provenance import has_synthetic_ancestry
    if not isinstance(metadata,dict):
        raise ValueError(label+' provenance must be an object')
    if has_synthetic_ancestry(metadata):
        raise ValueError(label+' contains synthetic/mock provenance and cannot enter a real experiment')


def validate_inputs(config, *, fingerprint=False):
    """Validate local metadata and file availability without constructing models.

    Optional execution fingerprint hashes manifests/checkpoints/code and records
    image size/mtime metadata; it is not a cryptographic hash of all image bytes.
    """
    c=normalize_experiment_config(config)
    d=c['data']; split=d['split']
    info=inspect_manifest(d['manifest'],d['image_root'],require_files=True)
    if info['num_samples'] != split['train_size']+split['test_size']:
        raise ValueError('Manifest size must equal the declared complete train/test split')
    if split['train_size']<2 or c['runtime']['batch_size']<2:
        raise ValueError('Training needs at least two examples and batch_size>=2')
    if split['path']:
        train,test=load_explicit_split(split['path'],info['image_ids'],train_size=split['train_size'],test_size=split['test_size'])
        split_source='explicit_id_file'
    else:
        train,test=deterministic_split(info['image_ids'],split['train_size'],split['test_size'],seed=split['seed'])
        split_source='seeded_sorted_id_reconstruction_not_verified_historical_split'
    unlabeled=None
    if d['unlabeled_manifest']:
        unlabeled=inspect_manifest(d['unlabeled_manifest'],d['unlabeled_root'],labeled=False,require_files=True)
        labeled_paths={(Path(d['image_root'])/r['image']).resolve() for r in info['samples']}
        unlabel_paths={(Path(d['unlabeled_root'])/r['image']).resolve() for r in unlabeled['samples']}
        if labeled_paths & unlabel_paths:
            raise ValueError('Post-training and labeled evaluation corpora must be disjoint local image paths')
        if c['runtime']['profile']=='fixture' and unlabeled['num_samples']>16:
            raise ValueError('Fixture unlabeled corpus is bounded to 16 images')
    for model in c['models']:
        validate_backbone_spec(model['backbone'],check_files=True)
        if c['runtime']['profile']=='experiment': _reject_synthetic(model['provenance'],model['name'])
    for label,path in _asset_paths(c):
        if not Path(path).exists(): raise FileNotFoundError(f'Missing local {label}: {path}')
    result={'manifest':{k:v for k,v in info.items() if k!='samples'},
            'train_indices':train,'test_indices':test,'split_source':split_source,
            'train_image_ids':[info['image_ids'][i] for i in train],
            'test_image_ids':[info['image_ids'][i] for i in test],
            'unlabeled_manifest':None if unlabeled is None else {k:v for k,v in unlabeled.items() if k!='samples'}}
    if fingerprint:
        files={}
        for label,path in _asset_paths(c):
            asset=Path(path)
            if asset.is_file(): files[str(asset)]=checkpoint_sha256(asset)
            elif label.endswith('repo_path'):
                files.update({str(p):checkpoint_sha256(p) for p in sorted(asset.rglob('*.py')) if '.git' not in p.parts})
            elif label.endswith('model_path'):
                files.update({str(p):checkpoint_sha256(p) for p in sorted(asset.rglob('*')) if p.is_file()})
        stats=[]
        for source in (info,unlabeled):
            if source is None: continue
            for row in source['samples']:
                path=Path(source['image_root'])/row['image']; stat=path.stat()
                stats.append([str(path.resolve()),stat.st_size,stat.st_mtime_ns])
        code_root=Path(__file__).resolve().parents[1]
        source_files={str(p.relative_to(code_root)):checkpoint_sha256(p)
                      for folder in ('configs','data','models','probing','training','perturbations','metrics','evaluation','report','scripts')
                      for p in sorted((code_root/folder).glob('*.py'))}
        result['input_hashes']=files
        result['image_stat_fingerprint']=hashlib.sha256(json.dumps(stats).encode()).hexdigest()
        result['source_hashes']=source_files
        result['run_fingerprint']=config_digest({'config':c,'inputs':result})
    return result


def _probe_config(c,protocol):
    p=c['probe']
    return ProbeConfig(protocol=protocol,num_classes=15,epochs=p[protocol+'_epochs'],
                       **{k:p[k] for k in ('hidden_dim','dropout','head_lr','encoder_lr','momentum','weight_decay')})


def _job_dir(output,name,phase):
    key=hashlib.sha256(name.encode()).hexdigest()[:16]
    path=output/'jobs'/key/phase
    for parent in [path,*path.parents]:
        if parent==output.parent: break
        if parent.is_symlink(): raise ValueError('Output jobs may not follow symlinks')
    path.mkdir(parents=True,exist_ok=True)
    return path


def _checkpoint_metadata_guard(spec,real):
    if not real or not spec.get('checkpoint'): return
    path=spec['checkpoint']['path']
    if Path(path).suffix=='.safetensors':
        from safetensors import safe_open
        with safe_open(path,framework='pt',device='cpu') as stream:
            metadata=stream.metadata() or {}
        # Safetensors metadata values are strings; common flags must not evade
        # provenance checks merely because the format serializes booleans.
        decoded={}
        for key,value in metadata.items():
            try: decoded[key]=json.loads(value)
            except (ValueError,TypeError): decoded[key]=value
        _reject_synthetic(decoded,'checkpoint')
        return
    # Official MAE files can contain only this additional inert args container.
    # Legacy torch.save files do not support mmap; both paths remain weights_only.
    from argparse import Namespace
    with torch.serialization.safe_globals([Namespace]):
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=zipfile.is_zipfile(path))
    if isinstance(payload,dict) and 'metadata' in payload:
        _reject_synthetic(payload['metadata'],'checkpoint')
    del payload


def _validate_output_tree(output):
    """Managed output descendants must not redirect writes outside this run."""
    for name in ('run_state.json','CURRENT.json','jobs','post_training','publications'):
        path=output/name
        if path.is_symlink(): raise ValueError('Managed output paths may not be symlinks')
        if path.is_dir():
            pending=[path]
            while pending:
                for item in pending.pop().iterdir():
                    if item.is_symlink(): raise ValueError('Managed output descendants may not be symlinks')
                    if item.is_dir(): pending.append(item)


def execute_experiment(config, *, resume=False, eval_only=False):
    c=normalize_experiment_config(config)
    from training.output import process_file_lock
    output=Path(c['runtime']['output_dir'])
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError('Experiment output cannot follow symlinks')
    output.parent.mkdir(parents=True,exist_ok=True)
    with process_file_lock(output.parent/('.'+output.name+'.experiment.lock')):
        return _execute_experiment(c,resume=resume,eval_only=eval_only)


def _execute_experiment(config, *, resume=False, eval_only=False):
    c=normalize_experiment_config(config)
    if resume and eval_only: raise ValueError('Choose resume or eval-only')
    inputs=validate_inputs(c,fingerprint=True)
    r=c['runtime']; output=Path(r['output_dir'])
    source_root=Path(__file__).resolve().parents[1]
    if output in (source_root,*source_root.parents) or output.is_symlink():
        raise ValueError('Use a dedicated output directory')
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError('Output parents must not be symlinks')
    output.mkdir(parents=True,exist_ok=True)
    _validate_output_tree(output)
    state_path=output/'run_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if not (resume or eval_only): raise ValueError('Output already exists; choose explicit resume/eval-only or a new output path')
        if state.get('run_fingerprint')!=inputs['run_fingerprint']:
            raise ValueError('Config, source or input provenance changed; use a new output directory')
    elif resume or eval_only:
        raise ValueError('Cannot resume/evaluate without a matching run_state.json')
    elif any(output.iterdir()):
        raise ValueError('Refusing to write into a nonempty unrecognized output directory')
    state={'run_fingerprint':inputs['run_fingerprint'],'status':'running','config':c,
           'inputs':inputs,'mode':'eval_only' if eval_only else 'resume' if resume else 'execute'}
    try:
        _atomic_json(state_path,state)
        results=_execute(c,inputs,output,resume=resume,eval_only=eval_only)
        # Tables and JSON live in one versioned publication directory. CURRENT
        # is changed last; readers never need a partially written aggregate.
        published=output/'publications'
        published.mkdir(exist_ok=True)
        publication=Path(tempfile.mkdtemp(prefix='run-',dir=published))
        _atomic_json(publication/'results.json',results)
        build_tables(results,publication/'tables')
        if c['report']['figures']:
            from report.build_figures import build_figures
            build_figures(results,publication/'figures')
        _atomic_json(output/'CURRENT.json',{'publication':str(publication),
            'results':str(publication/'results.json'),'tables':str(publication/'tables'/'INDEX.md'),
            'figures':str(publication/'figures'/'INDEX.md') if c['report']['figures'] else None,
            'run_fingerprint':inputs['run_fingerprint']})
        state['status']='complete'; state['publication']=str(publication)
        _atomic_json(state_path,state)
        return results
    except BaseException as exc:
        state['status']='failed'; state['error_type']=type(exc).__name__; state['error']=str(exc)
        _atomic_json(state_path,state)
        raise


def _execute(c,inputs,output,*,resume,eval_only):
    r=c['runtime']; d=c['data']; seed=r['seed']; device=r['device']
    torch.set_num_threads(r['num_threads']); torch.manual_seed(seed)
    real=r['profile']=='experiment'; evidence='measured_local' if real else 'synthetic_smoke'
    dataset=LocalManifestCXRDataset(d['manifest'],d['image_root'],d['image_size'],intensity=d['intensity'])
    train=Subset(dataset,inputs['train_indices']); test=Subset(dataset,inputs['test_indices'])
    train_labels=dataset.labels_tensor()[inputs['train_indices']]
    def loaders():
        sampler_rng=torch.Generator().manual_seed(seed)
        worker_rng=torch.Generator().manual_seed(seed+1)
        train_loader=DataLoader(train,batch_sampler=CompleteBatchSampler(len(train),r['batch_size'],sampler_rng),
            num_workers=r['num_workers'],collate_fn=collate_cxr,generator=worker_rng,persistent_workers=False)
        test_loader=DataLoader(test,batch_size=r['batch_size'],shuffle=False,num_workers=r['num_workers'],
            collate_fn=collate_cxr,generator=torch.Generator().manual_seed(seed+2))
        return train_loader,DeviceBatches(test_loader,device),{'sampler':sampler_rng,'worker':worker_rng}
    specs=configured_perturbations(c)
    if c['mitigation']['enabled'] or c['nca']['enabled']:
        present={spec.severity for spec in specs if spec.name=='gaussian_noise'}
        specs.extend(PerturbationSpec('gaussian_noise',sigma,{'sigma':sigma})
                     for sigma in c['mitigation']['noise_sigmas'] if sigma not in present)
    noise_specs=[PerturbationSpec('gaussian_noise',sigma,{'sigma':sigma}) for sigma in c['mitigation']['noise_sigmas']]
    results={'metadata':{'evidence':evidence,'real_experiments_run':real,'paper_metrics_reproduced':False,
            'downloads_performed':False,'device':device,'seed':seed,'config':c,'versions':_versions(),
            'inputs':inputs,'data_source':'local_manifest','class_names':dataset.class_names,
            'historical_split_verified':False,'normalization_and_unspecified_parameters':'explicit recorded implementation choices'},
            **{key:[] for key in ('robustness','lesion','lesion_details','training','post_training','mitigation','train_aug','nca','token_diagnostics','interventions')}}
    factories=[]
    for model in c['models']:
        _checkpoint_metadata_guard(model['backbone'],real)
        def factory(spec=model['backbone']):
            return build_backbone(spec)
        factories.append((model['name'],model['group'],factory,model['provenance']))
    if c['post_training']['enabled']:
        from scripts.post_training_jobs import execute_post_training, build_full_ijepa
        records=execute_post_training(c,output/'post_training',resume=resume,eval_only=eval_only)
        results['post_training']=records
        for record in records:
            def factory(record=record):
                full=build_full_ijepa(record['backbone'],registers=4 if record['variant']=='v5' else 0)
                load_training_model(record['checkpoint'],full)
                encoder=full.context_encoder
                spec=validate_backbone_spec({'backend':'native','family':'native','use_random_init':True,
                    'image_size':d['image_size'],'patch_size':record['backbone']['patch_size'],
                    'kwargs':{**{k:record['backbone'][k] for k in ('embed_dim','depth','num_heads')},
                              'num_register_tokens':encoder.num_register_tokens},
                    'normalization':record['normalization']})
                return BackboneAdapter(encoder,spec,{'source':'trained_native_post_checkpoint','checkpoint':record['checkpoint']})
            factories.append(('TexJEPA-'+record['variant'],'post',factory,record['metadata']))
    started=time.monotonic()
    for name,group,factory,provenance in factories:
        def fit(protocol,phase=None,augmentation=None):
            phase=phase or protocol
            job=_job_dir(output,name,phase); latest=job/'latest.pt'
            prior=resume_checkpoint(job) if eval_only or (resume and any(job.iterdir())) else None
            metadata={'run_fingerprint':inputs['run_fingerprint'],'phase':phase,'model':name,
                      'evidence':evidence,'mock':not real,'declared_source':provenance}
            torch.manual_seed(seed)
            encoder=factory()
            metadata['loaded_backbone']=encoder.provenance
            train_loader,test_batches,generators=loaders()
            pc=_probe_config(c,protocol)
            if eval_only:
                fitted=ProbeModel(encoder,pc).to(device)
                saved=load_training_model(prior,fitted,expected_metadata=metadata)
                if saved['epoch']!=pc.epochs: raise ValueError('Eval-only requires the completed probe epoch count')
                info={'checkpoint':str(prior),'completed_epochs':saved['epoch'],'mode':'eval_only'}
            else:
                fitted,info=fit_probe(encoder,train_loader,pc,device=device,training_labels=train_labels,
                    augmentation=augmentation,augmentation_seed=seed,checkpoint_dir=job,
                    resume_from=prior,metadata=metadata,generators=generators)
                if prior and not latest.exists(): restore_latest_pointer(job,info['checkpoint'])
            results['training'].append({'model':name,'phase':phase,'evidence':evidence,**info})
            return fitted,test_batches
        for protocol in c['protocols']:
            fitted,test_batches=fit(protocol)
            rows=evaluate_robustness(fitted.encoder,fitted.head,test_batches,specs,model_name=name,
                                    protocol=protocol,seed=seed,evidence=evidence)
            for row in rows: row.update(group=group,paper_model=name)
            results['robustness'].extend(rows)
            if protocol=='linear':
                _diagnostics(c,results,fitted,test_batches,test,name,group,evidence,device,seed)
            del fitted
        if 'linear' not in c['protocols']: continue
        if c['mitigation']['enabled'] and c['mitigation']['train_aug']:
            augmented,test_batches=fit('linear','linear_train_aug',
                lambda images,*,generator:gaussian_noise(images,c['mitigation']['train_aug_sigma'],generator=generator))
            records=evaluate_robustness(augmented.encoder,augmented.head,test_batches,noise_specs,
                model_name=name,protocol='linear_train_aug',seed=seed,evidence=evidence)
            results['train_aug'].append({'model':name,'evidence':evidence,'robustness':records})
            _interventions(results,name,'train_aug',records,evidence)
            del augmented
        if c['nca']['enabled']:
            base=factory(); base.to(device)
            probe=ProbeModel(base,_probe_config(c,'linear')).to(device)
            load_training_model(resume_checkpoint(_job_dir(output,name,'linear')),probe,
                                expected_metadata={'run_fingerprint':inputs['run_fingerprint'],'model':name,'phase':'linear'})
            n=c['nca']; model=NoiseConsistencyAdapter(probe.encoder,probe.encoder.embed_dim,num_classes=15,
                              hidden_dim=n['hidden_dim'],head=probe.head)
            job=_job_dir(output,name,'nca'); latest=job/'latest.pt'
            prior=resume_checkpoint(job) if eval_only or (resume and any(job.iterdir())) else None
            metadata={'run_fingerprint':inputs['run_fingerprint'],'model':name,'phase':'nca','evidence':evidence,'mock':not real}
            train_loader,test_batches,generators=loaders()
            if eval_only:
                model.to(device); saved=load_training_model(prior,model,expected_metadata=metadata)
                if saved['epoch']!=n['epochs']: raise ValueError('Eval-only requires completed NCA training')
                info={'checkpoint':str(prior),'completed_epochs':saved['epoch'],'mode':'eval_only'}
            else:
                model,info=train_nca_resumable(model,train_loader,
                    TrainingConfig(epochs=n['epochs'],learning_rate=n['learning_rate'],device=device,seed=seed),
                    noise_sigma=n['noise_sigma'],alignment_weight=n['alignment_weight'],
                    consistency_weight=n['consistency_weight'],supervised_weight=n['supervised_weight'],
                    training_labels=train_labels,checkpoint_dir=job,resume_from=prior,
                    metadata=metadata,generators=generators)
                if prior and not latest.exists(): restore_latest_pointer(job,info['checkpoint'])
            records=evaluate_robustness(_AdaptedEncoder(model),model.head,test_batches,noise_specs,
                        model_name=name,protocol='nca',seed=seed,evidence=evidence)
            results['nca'].append({'model':name,'evidence':evidence,'training':info,'robustness':records})
            _interventions(results,name,'noise_consistency_adapter',records,evidence)
            del model,probe,base
    results['metadata']['elapsed_seconds']=time.monotonic()-started
    return _jsonable(results)


def _interventions(results,name,method,records,evidence):
    clean=next(row for row in records if row['perturbation']=='clean')
    for row in records:
        if row['perturbation']!='gaussian_noise': continue
        original=next((r for r in results['robustness'] if r['model']==name and r['protocol']=='linear'
                    and r['perturbation']=='gaussian_noise' and r['severity']==row['severity']),None)
        results['interventions'].append({'model':name,'protocol':row['protocol'],'intervention':method,
            'noise_sigma':row['severity'],'clean_auroc':clean['auroc'],'noise_auroc':row['auroc'],
            'raw_drift':None if original is None else original['drift'],'adapted_drift':row['drift'],'evidence':evidence})


def _diagnostics(c,results,fitted,batches,test,name,group,evidence,device,seed):
    def samples():
        for sample in test:
            yield {**sample,'image':sample['image'].to(device)}
    if c['lesion']['enabled']:
        l=c['lesion']
        detail=evaluate_lesion_occlusion(fitted.encoder,fitted.head,samples(),model_name=name,protocol='linear',
            fill=l['fill'],controls=l['controls'],bootstrap_iterations=l['bootstrap_replicates'],seed=l['seed'],evidence=evidence)
        results['lesion'].append({**detail['overall'],'n_skipped':detail['overall']['n_skipped_records'],
                                 'group':group,'paper_model':name})
        results['lesion_details'].append({'model':name,'protocol':'linear','evidence':evidence,**detail})
    if c['mitigation']['enabled']:
        noise=[PerturbationSpec('gaussian_noise',s,{'sigma':s}) for s in c['mitigation']['noise_sigmas']]
        records=evaluate_mitigations(fitted.encoder,fitted.head,batches,noise,model_name=name,
                                    protocol='linear',seed=seed,evidence=evidence)
        results['mitigation'].extend(records)
        for method in ('none','median','gaussian'):
            _interventions(results,name,method,[row for row in records if row['mitigation']==method],evidence)
    # The paper's weak token saliency compares I-JEPA/MAE mean-patch readouts.
    # CLS/extra-head readouts cannot be replaced by patch means silently.
    sample=next((sample for sample in samples() if sample['boxes']),None)
    if sample is None:
        results['token_diagnostics'].append({'model':name,'protocol':'linear','evidence':evidence,'status':'no_annotated_held_out_image'})
        return
    images=sample['image'].unsqueeze(0); box=sample['boxes'][0]
    noisy=gaussian_noise(images,.05,generator=torch.Generator(device=device).manual_seed(seed))
    occluded=occlude(sample['image'],box['bbox'],fill=c['lesion']['fill']).unsqueeze(0)
    record={'model':name,'protocol':'linear','evidence':evidence,'image_id':sample['image_id'],
        'class_index':box['class_index'],'noise_sigma':.05,'lesion_bbox':box['bbox'],
        'noise':compare_token_drift(fitted.encoder,images,noisy),
        'lesion_occlusion':compare_token_drift(fitted.encoder,images,occluded)}
    if getattr(fitted.encoder,'spec',{}).get('pooling','mean_patch')=='mean_patch':
        record['saliency']=gradient_activation_saliency(fitted.encoder,fitted.head,images,box['class_index'])
    else:
        record['saliency']={'status':'not_applicable_to_non_mean_patch_readout',
                            'reason':'A patch-mean surrogate would change the actual trained classifier.'}
    results['token_diagnostics'].append(record)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--validate-inputs',action='store_true')
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--resume',action='store_true')
    mode.add_argument('--eval-only',action='store_true')
    args=parser.parse_args(argv); config=load_experiment_config(args.config)
    if args.validate_inputs: result=validate_inputs(config)
    elif args.execute or args.resume or args.eval_only:
        measured=execute_experiment(config,resume=args.resume,eval_only=args.eval_only)
        result={'status':'complete','evidence':measured['metadata']['evidence'],
                'current':str(Path(config['runtime']['output_dir'])/'CURRENT.json')}
    else: result=plan_experiment(config)
    print(json.dumps(_jsonable(result),indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__': main()
