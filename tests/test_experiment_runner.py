"""Real orchestration exercised with at most eight synthetic 32px CPU images."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from configs.experiment import normalize_experiment_config
from scripts.run_experiments import (CompleteBatchSampler, plan_experiment, validate_inputs,
                                     execute_experiment, _reject_synthetic)


def fixture_config(tmp_path):
    root=tmp_path/'images'; root.mkdir()
    records=[]
    for i in range(8):
        Image.fromarray(np.random.default_rng(i).integers(0,256,(32,32),dtype=np.uint8)).save(root/f'{i}.png')
        labels=[(i+j)%2 for j in range(15)]
        positive=labels.index(1)
        records.append({'image_id':str(i),'image':f'{i}.png','labels':labels,
                        'boxes':[{'class_index':positive,'bbox':[4,4,8,8]}]})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'class_names':[f'c{j}' for j in range(15)],'samples':records}))
    split=tmp_path/'split.json'
    split.write_text(json.dumps({'train_ids':['0','1','2','3'],'test_ids':['4','5','6','7']}))
    return normalize_experiment_config({'schema_version':1,
        'runtime':{'profile':'fixture','device':'cpu','batch_size':2,'output_dir':str(tmp_path/'output')},
        'data':{'manifest':str(manifest),'image_root':str(root),'image_size':32,
                'split':{'path':str(split),'train_size':4,'test_size':4}},
        'models':[{'name':'native fixture','backbone':{'backend':'native','use_random_init':True},
                   'provenance':{'source':'fixture','mock':True}}],
        'probe':{'linear_epochs':1,'mlp_epochs':1,'partial_ft_epochs':1,'hidden_dim':32},
        'report':{'figures':False},'nca':{'epochs':1,'hidden_dim':16},'lesion':{'bootstrap_replicates':5}})


def test_default_plan_and_input_validation_never_construct_or_decode(tmp_path,monkeypatch):
    c=fixture_config(tmp_path)
    monkeypatch.setattr(Image,'open',lambda *a,**k:pytest.fail('plan decoded pixels'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('plan loaded weights'))
    monkeypatch.setattr('scripts.run_experiments.build_backbone',lambda *a,**k:pytest.fail('plan built model'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('plan probed GPU'))
    assert plan_experiment(c)['execution_started'] is False
    inputs=validate_inputs(c)
    assert inputs['train_image_ids']==['0','1','2','3']
    assert inputs['test_image_ids']==['4','5','6','7']
    assert not Path(c['runtime']['output_dir']).exists()


@pytest.mark.parametrize('n,b',[(2,4),(3,2),(5,2),(7,3),(9,4),(13,4)])
def test_complete_batches_preserve_all_samples_without_singletons(n,b):
    sampler=CompleteBatchSampler(n,b,torch.Generator().manual_seed(42))
    chunks=list(sampler)
    assert sorted(x for batch in chunks for x in batch)==list(range(n))
    assert min(map(len,chunks))>=2 and len(chunks)==len(sampler)


def test_all_protocols_diagnostics_checkpoints_eval_and_resume(tmp_path):
    c=fixture_config(tmp_path)
    result=execute_experiment(c)
    assert result['metadata']['evidence']=='synthetic_smoke'
    assert result['metadata']['real_experiments_run'] is False
    assert {r['protocol'] for r in result['robustness']}=={'linear','mlp','partial_ft'}
    assert {r['intervention'] for r in result['interventions']}=={'none','median','gaussian','train_aug','noise_consistency_adapter'}
    assert {r['noise_sigma'] for r in result['interventions']}=={.05,.1}
    assert len(result['token_diagnostics'])==1
    output=Path(c['runtime']['output_dir'])
    pointer=json.loads((output/'CURRENT.json').read_text())
    assert Path(pointer['results']).is_file() and Path(pointer['tables']).is_file()
    assert len(list(output.glob('jobs/*/*/latest.pt')))==5
    with pytest.raises(ValueError,match='already exists'): execute_experiment(c)
    evaluated=execute_experiment(c,eval_only=True)
    assert evaluated['robustness']==result['robustness']
    resumed=execute_experiment(c,resume=True)
    assert resumed['robustness']==result['robustness']
    changed=copy.deepcopy(c); changed['probe']['head_lr']*=2
    with pytest.raises(ValueError,match='provenance changed'): execute_experiment(changed,resume=True)


def test_failed_evaluation_keeps_last_complete_publication(tmp_path):
    c=fixture_config(tmp_path); c['protocols']=['linear']; c['nca']['enabled']=False
    c['mitigation']['train_aug']=False
    execute_experiment(c)
    output=Path(c['runtime']['output_dir']); previous=(output/'CURRENT.json').read_bytes()
    for checkpoint in output.glob('jobs/*/linear/*.pt'): checkpoint.unlink()
    with pytest.raises((ValueError,FileNotFoundError)): execute_experiment(c,eval_only=True)
    assert (output/'CURRENT.json').read_bytes()==previous
    assert json.loads((output/'run_state.json').read_text())['status']=='failed'
    assert not (output/'.execution.lock').exists()


def test_synthetic_cannot_be_relabelled_real_or_bypass_fixture_bound(tmp_path):
    c=fixture_config(tmp_path)
    c['runtime']['profile']='experiment'
    with pytest.raises(ValueError,match='random-init'): normalize_experiment_config(c)
    with pytest.raises(ValueError,match='synthetic/mock'):
        _reject_synthetic({'source':'author','parent':{'epoch_is_lineage_label':True}},'x')
    c['runtime']['profile']='fixture'; c['models'][0]['backbone']['kwargs']['depth']=32
    with pytest.raises(ValueError,match='bounded native'): normalize_experiment_config(c)


def test_unlabeled_corpus_overlap_rejected_before_pixel_reads(tmp_path):
    c=fixture_config(tmp_path)
    unlabeled=tmp_path/'u.json'; unlabeled.write_text(json.dumps({'samples':[{'image_id':'renamed','image':'0.png'}]}))
    c['data']['unlabeled_manifest']=str(unlabeled); c['data']['unlabeled_root']=c['data']['image_root']
    with pytest.raises(ValueError,match='disjoint'): validate_inputs(c)


@pytest.mark.parametrize('section,key,value',[('probe','hidden_dim',100000000),('nca','hidden_dim',100000000),
                                           ('runtime','num_workers',1000),('runtime','num_threads',1000)])
def test_fixture_rejects_unbounded_allocation_before_execution(tmp_path,section,key,value):
    c=fixture_config(tmp_path); c[section][key]=value
    with pytest.raises(ValueError,match='bounded'): normalize_experiment_config(c)


def test_partial_post_objective_overrides_are_returned_complete(tmp_path):
    c=fixture_config(tmp_path); c['post_training']['objective']={'gaussian_sigma':.12}
    normalized=normalize_experiment_config(c)
    assert normalized['post_training']['objective']['gaussian_sigma']==.12
    assert normalized['post_training']['objective']['lambda_cov']==.04


@pytest.mark.parametrize('child',['publications','post_training','jobs','run_state.json'])
def test_managed_symlinks_are_rejected_before_any_write(tmp_path,child,monkeypatch):
    c=fixture_config(tmp_path); output=Path(c['runtime']['output_dir']); output.mkdir()
    outside=tmp_path/'outside'; outside.mkdir()
    (output/child).symlink_to(outside,target_is_directory=True)
    monkeypatch.setattr('scripts.run_experiments._execute',lambda *a,**k:pytest.fail('execution began'))
    with pytest.raises(ValueError,match='symlinks'): execute_experiment(c)
    assert not list(outside.iterdir())


def test_checkpoint_guard_accepts_safe_mae_namespace_and_legacy_format(tmp_path):
    from argparse import Namespace
    from scripts.run_experiments import _checkpoint_metadata_guard
    for zip_format in (True,False):
        path=tmp_path/f'weights_{zip_format}.pt'
        torch.save({'model':{'x':torch.ones(1)},'args':Namespace(epochs=300),
                    'metadata':{'source':'declared_local'}},path,_use_new_zipfile_serialization=zip_format)
        _checkpoint_metadata_guard({'checkpoint':{'path':str(path)}},True)
    torch.save({'metadata':{'parent':{'mock':True}}},path)
    with pytest.raises(ValueError,match='synthetic/mock'):
        _checkpoint_metadata_guard({'checkpoint':{'path':str(path)}},True)


@pytest.mark.parametrize('setting',['noise_sigmas','target_blocks'])
def test_fixture_diagnostic_multipliers_are_bounded(tmp_path,setting):
    c=fixture_config(tmp_path)
    if setting=='noise_sigmas': c['mitigation'][setting]=list(range(100))
    else: c['post_training']['objective'][setting]=100000000
    with pytest.raises(ValueError,match='bounded'): normalize_experiment_config(c)


def test_nca_sigma_baselines_exist_even_if_main_noise_list_is_empty(tmp_path):
    c=fixture_config(tmp_path);c['protocols']=['linear'];c['perturbations']['gaussian_noise']=[]
    c['mitigation']['enabled']=False
    result=execute_experiment(c)
    nca=[row for row in result['interventions'] if row['intervention']=='noise_consistency_adapter']
    assert len(nca)==2 and all(row['raw_drift'] is not None for row in nca)



def test_post_training_outputs_flow_into_all_downstream_diagnostics(tmp_path):
    import importlib.util
    path=Path(__file__).with_name('test_post_training_jobs.py')
    spec=importlib.util.spec_from_file_location('post_fixture_helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    c=fixture_config(tmp_path);pretrain_root=tmp_path/'separate_unlabeled';pretrain_root.mkdir()
    post=helper.fixture_config(pretrain_root)
    c['post_training']=post['post_training']
    c['data'].update(unlabeled_manifest=post['data']['unlabeled_manifest'],unlabeled_root=post['data']['unlabeled_root'])
    c['protocols']=['linear'];c['nca']['enabled']=False;c['mitigation']['enabled']=False
    c['lesion']['enabled']=False
    result=execute_experiment(c)
    assert {r['model'] for r in result['robustness']}=={'native fixture','TexJEPA-v4','TexJEPA-v5','TexJEPA-v6'}
    assert all(row['metadata']['epoch']==2 for row in result['post_training'])
    for path in Path(c['runtime']['output_dir']).glob('jobs/*/linear/latest.pt'):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['metadata']['model']=='TexJEPA-v5':
            assert payload['metadata']['loaded_backbone']['backbone_spec']['kwargs']['num_register_tokens']==4
    evaluated=execute_experiment(c,eval_only=True)
    assert evaluated['robustness']==result['robustness']



def test_output_alias_does_not_hide_symlink_from_boundary_check(tmp_path):
    c=fixture_config(tmp_path);target=tmp_path/'other';target.mkdir()
    Path(c['runtime']['output_dir']).symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinks'):execute_experiment(c)
    assert not list(target.iterdir())



def test_resume_orphan_probes_and_nca_does_not_repeat_training(tmp_path,monkeypatch):
    c=fixture_config(tmp_path);c['protocols']=['linear'];c['mitigation']['enabled']=False
    execute_experiment(c)
    output=Path(c['runtime']['output_dir'])
    epochs={p:p.stat().st_mtime_ns for p in output.glob('jobs/*/*/epoch_0001.pt')}
    for path in output.glob('jobs/*/*/latest.pt'):path.unlink()
    def forbidden(*a,**k):pytest.fail('completed orphan epoch repeated optimizer steps')
    monkeypatch.setattr(torch.optim.SGD,'step',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    execute_experiment(c,resume=True)
    assert all(path.stat().st_mtime_ns==stamp for path,stamp in epochs.items())
    assert len(list(output.glob('jobs/*/*/latest.pt')))==2
