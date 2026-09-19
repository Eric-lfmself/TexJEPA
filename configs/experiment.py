"""Explicit offline execution contract for the full TexJEPA experiment suite.

Unlike the bounded synthetic profile, this configuration describes real jobs.
Loading and planning never instantiate a model, decode an image or use a GPU.
Paper-omitted optimization details remain explicit, recorded implementation choices.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .schema import load_config


def _number(value, name, minimum=0, *, integer=False, positive=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < minimum or (positive and value == 0)):
        raise ValueError(f'{name}: invalid numeric value')


def _path(value, base, name, *, optional=False, resolve=True):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or '://' in value or '\x00' in value:
        raise ValueError(f'{name} must be an explicit local path')
    path = Path(value).expanduser()
    return str((base/path).resolve()) if resolve else os.path.abspath(base/path)


def _merge(base, updates, section):
    if not isinstance(updates, dict):
        raise ValueError(f'{section} must be an object')
    unknown = set(updates)-set(base)
    if unknown:
        raise ValueError(f'{section}: unknown settings {sorted(unknown)}')
    result = copy.deepcopy(base)
    result.update(copy.deepcopy(updates))
    return result


def load_experiment_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    return normalize_experiment_config(raw, base=path.parent)


def normalize_experiment_config(raw, *, base=None):
    base = Path(base or Path.cwd()).resolve()
    if not isinstance(raw, dict) or type(raw.get('schema_version')) is not int or raw.get('schema_version') != 1:
        raise ValueError('Expected experiment schema_version=1')
    allowed = {'schema_version','runtime','data','models','protocols','probe','perturbations',
               'lesion','mitigation','nca','post_training','report'}
    if set(raw)-allowed:
        raise ValueError(f'Unknown experiment sections: {sorted(set(raw)-allowed)}')
    paper = load_config('paper')
    c = {'schema_version': 1}
    c['report']=_merge({'figures':True},raw.get('report',{}),'report')
    if type(c['report']['figures']) is not bool: raise ValueError('report.figures must be boolean')
    c['runtime'] = _merge({'device':'cuda:0','profile':'experiment','batch_size':32,
                          'num_workers':0,'seed':42,'num_threads':1,
                          'output_dir':'../outputs/experiment'}, raw.get('runtime',{}), 'runtime')
    r = c['runtime']
    if r['profile'] not in ('experiment','fixture') or not (isinstance(r['device'],str) and re.fullmatch(r'cpu|cuda:[0-9]+',r['device'])):
        raise ValueError('Use an explicit cpu/cuda device and experiment/fixture profile; automatic selection is not supported')
    for key in ('batch_size','num_threads'):
        _number(r[key], 'runtime.'+key, integer=True, positive=True)
    _number(r['num_workers'],'runtime.num_workers',integer=True)
    _number(r['seed'],'runtime.seed',integer=True)
    r['output_dir'] = _path(r['output_dir'],base,'runtime.output_dir',resolve=False)
    c['data'] = _merge({'manifest':None,'image_root':None,'image_size':224,
        'intensity':{'policy':'uint8','scale':None},
        'split':{'path':None,'train_size':12000,'test_size':3000,'seed':42},
        'unlabeled_manifest':None,'unlabeled_root':None},raw.get('data',{}),'data')
    d = c['data']
    for key in ('manifest','image_root'):
        d[key] = _path(d[key],base,'data.'+key)
    for key in ('unlabeled_manifest','unlabeled_root'):
        d[key] = _path(d[key],base,'data.'+key,optional=True)
    if bool(d['unlabeled_manifest']) != bool(d['unlabeled_root']):
        raise ValueError('Unlabeled manifest and root must be supplied together')
    _number(d['image_size'],'data.image_size',integer=True,positive=True)
    d['intensity'] = _merge({'policy':'uint8','scale':None},d['intensity'],'data.intensity')
    if d['intensity']['policy'] not in ('uint8','uint16_scale'):
        raise ValueError('Unknown image intensity policy')
    if d['intensity']['policy'] == 'uint16_scale':
        _number(d['intensity']['scale'],'data.intensity.scale',positive=True)
        if d['intensity']['scale'] > 65535:
            raise ValueError('uint16 scale cannot exceed 65535')
    elif d['intensity']['scale'] is not None:
        raise ValueError('uint8 intensity does not accept a scale')
    d['split'] = _merge({'path':None,'train_size':12000,'test_size':3000,'seed':42},d['split'],'data.split')
    d['split']['path'] = _path(d['split']['path'],base,'data.split.path',optional=True)
    for key in ('train_size','test_size'):
        _number(d['split'][key],'data.split.'+key,integer=True,positive=True)
    _number(d['split']['seed'],'data.split.seed',integer=True)
    c['models'] = copy.deepcopy(raw.get('models',[]))
    if not isinstance(c['models'],list) or not c['models']:
        raise ValueError('Declare at least one locally supplied model')
    names=[]
    for model in c['models']:
        if not isinstance(model,dict) or set(model)-{'name','backbone','provenance','group'}:
            raise ValueError('Each model declares name, backbone and provenance')
        name=model.get('name')
        if not isinstance(name,str) or not name.strip():
            raise ValueError('Model names must be nonempty')
        names.append(name)
        model.setdefault('group','main')
        if model['group'] not in ('main','historical','external','post'):
            raise ValueError('Invalid model table group')
        if not isinstance(model.get('backbone'),dict) or not isinstance(model.get('provenance'),dict):
            raise ValueError('Model backbone and declared provenance are required')
        # Backends own their complete schema; resolve only explicitly local paths.
        spec=model['backbone']
        if isinstance(spec.get('checkpoint'), dict):
            spec['checkpoint']['path']=_path(spec['checkpoint'].get('path'),base,'backbone.checkpoint.path')
        for key in ('checkpoint','model_path','repo_path'):
            if key in spec and spec[key] is not None and not isinstance(spec[key],dict):
                spec[key]=_path(spec[key],base,'backbone.'+key)
        if r['profile']=='experiment' and (not isinstance(model['provenance'].get('source'),str) or not model['provenance']['source'].strip()):
            raise ValueError('Real models require a declared source; strict loading does not establish training history')
    from models.backbones import validate_backbone_spec
    for model in c['models']:
        spec=model['backbone']=validate_backbone_spec(model['backbone'])
        if spec['image_size']!=d['image_size'] or spec['in_channels']!=3:
            raise ValueError('All backbones must use the common image size and RGB data contract')
        if r['profile']=='experiment' and spec['use_random_init']:
            raise ValueError('Real experiment profiles require local pretrained assets, never random-init comparisons')
        if r['profile']=='fixture':
            kw=spec['kwargs']
            if (spec['backend']!='native' or kw['embed_dim']>64 or kw['depth']>2 or kw['num_heads']>4 or spec['patch_size']<8 or kw['num_register_tokens']>4):
                raise ValueError('Fixture backbones must be bounded native mini models')
    if len(names)!=len(set(names)):
        raise ValueError('Model names must be unique')
    c['protocols']=copy.deepcopy(raw.get('protocols',['linear','mlp','partial_ft']))
    if (not isinstance(c['protocols'],list) or not c['protocols']
            or len(set(c['protocols']))!=len(c['protocols'])
            or set(c['protocols'])-{'linear','mlp','partial_ft'}):
        raise ValueError('Choose unique linear/mlp/partial_ft protocols')
    c['probe']=_merge({key:paper['probing'][key] for key in ('linear_epochs','mlp_epochs','partial_ft_epochs',
                    'hidden_dim','dropout','head_lr','encoder_lr','momentum','weight_decay')},raw.get('probe',{}),'probe')
    p=c['probe']
    for key in ('linear_epochs','mlp_epochs','partial_ft_epochs','hidden_dim'):
        _number(p[key],'probe.'+key,integer=True,positive=True)
    for key in ('head_lr','encoder_lr'):
        _number(p[key],'probe.'+key,positive=True)
    if p['encoder_lr']>=p['head_lr']:
        raise ValueError('Partial FT uses a lower encoder LR than head LR')
    for key in ('momentum','weight_decay','dropout'):
        _number(p[key],'probe.'+key)
    if p['dropout']>=1:
        raise ValueError('Dropout must be less than 1')
    c['perturbations']=_merge(paper['perturbations'],raw.get('perturbations',{}),'perturbations')
    c['lesion']=_merge(dict(paper['lesion'],enabled=True),raw.get('lesion',{}),'lesion')
    c['mitigation']=_merge({'enabled':True,'noise_sigmas':[.05,.10],'train_aug':True,
                          'train_aug_sigma':.05},raw.get('mitigation',{}),'mitigation')
    c['nca']=_merge({'enabled':True,'epochs':15,'hidden_dim':64,'learning_rate':.001,
                   'noise_sigma':.05,'alignment_weight':1.,'consistency_weight':1.,
                   'supervised_weight':1.},raw.get('nca',{}),'nca')
    c['post_training']=_merge({'enabled':False,'variants':['v4','v5','v6'],
        'baseline_checkpoint':None,'parent_checkpoints':{},'backbone':dict(paper['model']),
        'epochs':{'v4':50,'v5':300,'v6':300},'training':{},
        'normalization':{'mean':[0.,0.,0.],'std':[1.,1.,1.]},
        'objective':dict(paper['post_training'])},raw.get('post_training',{}),'post_training')
    post=c['post_training']
    post['backbone']=_merge(paper['model'],post['backbone'],'post_training.backbone')
    post['objective']=_merge(paper['post_training'],post['objective'],'post_training.objective')
    post['baseline_checkpoint']=_path(post['baseline_checkpoint'],base,'post_training.baseline_checkpoint',optional=True)
    if not isinstance(post['parent_checkpoints'],dict) or set(post['parent_checkpoints'])-{'v4','v5','v6'}:
        raise ValueError('Invalid post-training parents')
    post['parent_checkpoints']={k:_path(v,base,'post_training.parent_checkpoints.'+k)
                                for k,v in post['parent_checkpoints'].items()}
    if not isinstance(post['variants'],list) or len(set(post['variants']))!=len(post['variants']) or set(post['variants'])-{'v4','v5','v6'}:
        raise ValueError('Invalid post-training variants')
    post['epochs']=_merge({'v4':50,'v5':300,'v6':300},post['epochs'],'post_training.epochs')
    for variant,epochs in post['epochs'].items():
        _number(epochs,'post_training.epochs.'+variant,integer=True,positive=True)
    if not isinstance(post['training'],dict):
        raise ValueError('post_training.training must be an object')
    if set(post['training'])-{'learning_rate','weight_decay','checkpoint_every','scheduler','ema_start','ema_end'}:
        raise ValueError('Unsupported post_training training option')
    from training import TrainingConfig
    TrainingConfig(**post['training'])
    norm=post['normalization']
    if not isinstance(norm,dict) or set(norm)!={'mean','std'}:
        raise ValueError('Post-training normalization requires mean/std')
    for key in ('mean','std'):
        if not isinstance(norm[key],list) or len(norm[key])!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or (key=='std' and v<=0) for v in norm[key]):
            raise ValueError('Invalid post-training normalization')
    if post['enabled']:
        if any('TexJEPA-'+v in names for v in post['variants']):
            raise ValueError('Declared model name collides with generated post-training variant')
        if not post['variants'] or not d['unlabeled_manifest']:
            raise ValueError('Post-training requires variants and a separate unlabeled corpus')
        if 'v4' in post['variants'] and not (post['baseline_checkpoint'] or post['parent_checkpoints'].get('v4')):
            raise ValueError('v4 requires a local full I-JEPA v3.1/201 checkpoint')
        for variant in ('v5','v6'):
            if variant in post['variants'] and variant not in post['parent_checkpoints']:
                if 'v4' not in post['variants'] or post['epochs']['v4'] < 50:
                    raise ValueError('v5/v6 require an actual v4 epoch-50 checkpoint or a v4 run reaching epoch 50')
    for section,key in [('lesion','enabled'),('mitigation','enabled'),('mitigation','train_aug'),('nca','enabled'),('post_training','enabled')]:
        if type(c[section][key]) is not bool:
            raise ValueError(f'{section}.{key} must be boolean')
    if r['profile']=='fixture':
        if p['hidden_dim']>512 or c['nca']['hidden_dim']>64 or r['num_workers']!=0 or r['num_threads']>4 or len(c['models'])>12:
            raise ValueError('Fixture heads, workers, threads and model count must remain bounded')
        if (c['lesion']['bootstrap_replicates']>2000 or c['lesion']['controls']>5
                or len(c['mitigation']['noise_sigmas'])>8
                or max(post['objective']['target_blocks'],post['objective']['register_target_blocks'])>6
                or any(v>2 for v in c['perturbations']['gaussian_blur'])
                or sum(len(v) for v in c['perturbations'].values() if isinstance(v,list))>64):
            raise ValueError('Fixture diagnostic workload must remain bounded')
        if r['device']!='cpu' or not 2<=r['batch_size']<=4 or d['image_size'] not in (32,64):
            raise ValueError('Fixtures require CPU, batch2-4 and 32/64px images')
        if d['split']['train_size']+d['split']['test_size']>16:
            raise ValueError('Fixture dataset is bounded to 16 images')
        if any(p[k]>2 for k in ('linear_epochs','mlp_epochs','partial_ft_epochs')) or c['nca']['epochs']>2:
            raise ValueError('Fixture training is bounded to two epochs')
        if post['enabled'] and (post['backbone']['image_size']!=d['image_size'] or post['backbone']['embed_dim']>64 or post['backbone']['depth']>2 or post['backbone']['num_heads']>4 or post['backbone']['predictor_dim']>64 or post['backbone']['predictor_depth']>2 or post['backbone']['patch_size']<8 or post['backbone']['num_register_tokens']>4):
            raise ValueError('Fixture post-training requires matching bounded mini models')
        if post['enabled'] and any(post['epochs'][v]>2 for v in post['variants']):
            raise ValueError('Fixture post-training is bounded; supply labelled parent fixtures for branch tests')
    # Reuse the established scientific parameter validators without its CPU-only execution gate.
    checks={'perturbations':c['perturbations'], 'lesion':{k:v for k,v in c['lesion'].items() if k!='enabled'},
            'probing':dict(p,protocol='linear',smoke_epochs=1),
            'post_training':post['objective'], 'model':post['backbone']}
    load_config('paper',overrides=checks)
    if c['lesion']['confidence'] != .95:
        raise ValueError('The implemented lesion bootstrap uses confidence=0.95')
    if not isinstance(c['mitigation']['noise_sigmas'],list) or not c['mitigation']['noise_sigmas']:
        raise ValueError('Mitigation requires noise levels')
    for sigma in c['mitigation']['noise_sigmas']+[c['mitigation']['train_aug_sigma'],c['nca']['noise_sigma']]:
        _number(sigma,'noise sigma')
    if len(set(c['mitigation']['noise_sigmas']))!=len(c['mitigation']['noise_sigmas']):
        raise ValueError('Duplicate mitigation noise levels')
    for key in ('epochs','hidden_dim'):
        _number(c['nca'][key],'nca.'+key,integer=True,positive=True)
    _number(c['nca']['learning_rate'],'nca.learning_rate',positive=True)
    for key in ('alignment_weight','consistency_weight','supervised_weight'):
        _number(c['nca'][key],'nca.'+key)
    if not any(c['nca'][k]>0 for k in ('alignment_weight','consistency_weight','supervised_weight')):
        raise ValueError('NCA objective cannot be identically zero')
    return c


def config_digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
