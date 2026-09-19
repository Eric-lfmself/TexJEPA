"""Offline diagnostic figure exports from recorded results, without new measurements.

General diagnostic layouts, not a claim of pixel-identical manuscript figures or
publisher certification. Missing values remain gaps; lesion uncertainty is the
recorded image-cluster bootstrap 95% interval. Source JSON is retained unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#6D6D00','#222222','#56B4E9','#884488']
MARKERS=['o','s','^','D','v','P','X','>']


def _value(x):
    return float(x) if type(x) in (float,int) and math.isfinite(x) else float('nan')


def _sigma(value):
    if isinstance(value, bool):
        raise ValueError('Noise sigma must be finite and nonnegative')
    try:
        sigma = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Noise sigma must be finite and nonnegative') from exc
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Noise sigma must be finite and nonnegative')
    return sigma


def _severity(row):
    if row['perturbation'] == 'clean':
        return 0.0
    if row['perturbation'] == 'gaussian_noise':
        return _sigma(row['severity'])
    return row['severity']


def _validate_conditions(results):
    seen = set()
    for row in results.get('robustness', []):
        key = (row['model'], row['protocol'], row['perturbation'], _severity(row))
        if key in seen:
            raise ValueError(f'Duplicate robustness condition: {key}')
        seen.add(key)
    seen = set()
    for row in results.get('interventions', []):
        if row.get('noise_sigma') is None or 'noise_auroc' not in row:
            continue
        key = (row['model'], row.get('protocol', 'unspecified'),
               row['intervention'], _sigma(row['noise_sigma']))
        if key in seen:
            raise ValueError(f'Duplicate intervention condition: {key}')
        seen.add(key)


def _publish_directory(staging, output):
    """Atomically rename within one parent, failing if any destination exists."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        # macOS SDK sys/stdio.h: RENAME_EXCL = 0x00000004.
        function = getattr(library, 'renameatx_np', None)
        flags = 4
    elif sys.platform.startswith('linux'):
        # Linux include/uapi/linux/fs.h: RENAME_NOREPLACE = (1 << 0).
        function = getattr(library, 'renameat2', None)
        flags = 1
    else:
        function = None
    if function is None:
        raise OSError(errno.ENOTSUP, 'Atomic no-clobber directory publication is unavailable')
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                        ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if staging.parent != output.parent:
        raise ValueError('Publication requires staging and output in the same directory')
    parent_fd = os.open(output.parent, os.O_RDONLY)
    try:
        result = function(parent_fd, os.fsencode(staging.name), parent_fd,
                          os.fsencode(output.name), flags)
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(output))
    finally:
        os.close(parent_fd)


def _raw_baselines(results, model, rows, sigmas):
    """Keep recorded baselines separate by probe protocol; never infer scores."""
    baselines = {}
    for row in rows:
        protocol = row.get('protocol', 'unspecified')
        # These two runner phases explicitly start from the frozen linear probe.
        raw_protocol = {'linear_train_aug': 'linear', 'nca': 'linear'}.get(protocol, protocol)
        baseline = baselines.setdefault(raw_protocol, {'clean': None, 'noise': {}, 'drift': {}})
        if row.get('raw_drift') is not None:
            sigma = _sigma(row['noise_sigma'])
            value = _value(row['raw_drift'])
            if math.isfinite(value):
                previous = baseline['drift'].get(sigma)
                if previous is not None and previous != value:
                    raise ValueError('Conflicting recorded raw drift for one baseline condition')
                baseline['drift'][sigma] = value
    for row in results.get('robustness', []):
        if row['model'] != model or row['protocol'] not in baselines:
            continue
        baseline = baselines[row['protocol']]
        if row['perturbation'] == 'clean':
            baseline['clean'] = row.get('auroc')
        elif row['perturbation'] == 'gaussian_noise':
            sigma = _sigma(row['severity'])
            baseline['noise'][sigma] = row.get('auroc')
            value = _value(row.get('drift'))
            if math.isfinite(value):
                previous = baseline['drift'].get(sigma)
                if previous is not None and previous != value:
                    raise ValueError('Conflicting recorded raw drift for one baseline condition')
                baseline['drift'][sigma] = value
    return {protocol: ([ _value(values['clean']) for _ in sigmas ],
                       [ _value(values['noise'].get(sigma)) for sigma in sigmas ],
                       [ _value(values['drift'].get(sigma)) for sigma in sigmas ])
            for protocol, values in baselines.items()}


def build_figures(results,output_dir):
    """Create a new directory atomically; never overwrite existing publications."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    evidence=results.get('metadata',{}).get('evidence')
    if evidence not in ('synthetic_smoke','measured_local'):
        raise ValueError('Figures require explicit measured_local/synthetic_smoke provenance')
    for section in ('robustness','lesion','mitigation','interventions','token_diagnostics'):
        if any(row.get('evidence')!=evidence for row in results.get(section,[])):
            raise ValueError('Mixed figure evidence is forbidden')
    _validate_conditions(results)
    source_sha256=hashlib.sha256(json.dumps(results,sort_keys=True,allow_nan=False).encode()).hexdigest()
    output=Path(output_dir)
    if output.exists() or output.is_symlink(): raise FileExistsError('Use a new figure output directory')
    if any(parent.is_symlink() for parent in output.parents): raise ValueError('Figure output cannot follow symlinks')
    # Check the caller's path before normalizing so symlink aliases are rejected.
    output=Path(os.path.abspath(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    manifest={'evidence':evidence,'source_sha256':source_sha256,
              'matplotlib':matplotlib.__version__,'figure_layout':'general diagnostic export, not manuscript facsimile',
              'missing_values':'gaps/NR, never zeros','figures':[]}
    def save(fig,name,description,*,title=None):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Figure names must stay inside the publication directory')
        fig.suptitle((title or name.replace('_',' '))+'\n'+evidence,fontsize=11)
        files=[]
        for suffix in ('png','pdf','svg'):
            filename=name+'.'+suffix
            fig.savefig(staging/filename,dpi=160,facecolor='white')
            files.append(filename)
        plt.close(fig)
        manifest['figures'].append({'name':name,'files':files,'description':description})
    try:
        with plt.rc_context({'font.size':8,'axes.titlesize':9,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42}):
            grouped=defaultdict(list)
            for row in results.get('robustness',[]):
                if row['perturbation']!='clean': grouped[(row['protocol'],row['perturbation'])].append(row)
            for group_index,((protocol,kind),rows) in enumerate(grouped.items(),1):
                models=list(dict.fromkeys(r['model'] for r in rows))
                categorical=any(not isinstance(_severity(r),(int,float)) for r in rows)
                conditions=list(dict.fromkeys(_severity(r) for r in rows))
                if not categorical: conditions.sort()
                for page,start in enumerate(range(0,len(models),8),1):
                    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
                    for index,model in enumerate(models[start:start+8]):
                        mapping={_severity(r):r for r in rows if r['model']==model}
                        for ax,metric in zip(axes,('auroc','drift')):
                            x=list(range(len(conditions))) if categorical else conditions
                            y=[_value(mapping.get(condition,{}).get(metric)) for condition in conditions]
                            ax.plot(x,y,color=COLORS[index],marker=MARKERS[index],label=model,linewidth=1.25)
                            if categorical: ax.set_xticks(x,conditions,rotation=25,ha='right')
                            ax.set_xlabel(kind+' severity'); ax.set_ylabel('AUROC' if metric=='auroc' else '1 - cosine similarity')
                            ax.set_ylim((0,1) if metric=='auroc' else (0,2)); ax.grid(alpha=.2)
                    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
                    save(fig,f'robustness_{group_index}_{page}',
                         f'Protocol: {protocol}; perturbation: {kind}. Recorded AUROC and cosine drift by severity; no AUROC confidence intervals were estimated. Missing values remain line gaps. Source: robustness.',
                         title=f'{protocol} / {kind} / page {page}')
            lesions=results.get('lesion',[])
            for page,start in enumerate(range(0,len(lesions),12),1):
                rows=lesions[start:start+12]
                fig,ax=plt.subplots(figsize=(11,max(4,len(rows)*.5+1.8)),layout='constrained')
                for index,row in enumerate(rows):
                    mean,low,high=(_value(row.get(key)) for key in ('delta_lesion','ci_low','ci_high'))
                    if math.isfinite(mean): ax.plot(mean,index,'o',color=COLORS[0])
                    else: ax.text(.02,index,'NR',transform=ax.get_yaxis_transform(),va='center')
                    if math.isfinite(low) and math.isfinite(high): ax.plot([low,high],[index,index],color=COLORS[0])
                ax.set_yticks(range(len(rows)),[r['model']+f" (n={r.get('n_images','NR')})" for r in rows])
                ax.axvline(0,color='#777777',linewidth=.8); ax.set_xlabel('Class-aligned lesion logit change relative to controls; 95% image-cluster bootstrap CI')
                ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
                save(fig,f'lesion_occlusion_{page}','Points: recorded delta lesion. Lines: 95% image-cluster bootstrap CI. n is images; missing estimates are NR. Source: lesion.')
            interventions=defaultdict(list)
            for row in results.get('interventions',[]):
                if row.get('noise_sigma') is not None and 'noise_auroc' in row:
                    interventions[row['model']].append(row)
            for index,(model,rows) in enumerate(interventions.items(),1):
                methods=list(dict.fromkeys((r.get('protocol','unspecified'),r['intervention']) for r in rows)); sigmas=sorted({_sigma(r['noise_sigma']) for r in rows})
                fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
                for k,method in enumerate(methods):
                    selected={_sigma(r['noise_sigma']):r for r in rows if (r.get('protocol','unspecified'),r['intervention'])==method}
                    for ax,metric in zip(axes,('clean_auroc','noise_auroc','adapted_drift')):
                        ax.plot(sigmas,[_value(selected.get(s,{}).get(metric)) for s in sigmas],
                                marker=MARKERS[k%8],color=COLORS[k%8],label=method[1]+' / '+method[0])
                        ax.set_xlabel('Gaussian noise sigma'); ax.set_ylabel('Clean AUROC' if metric=='clean_auroc' else 'Noisy AUROC' if metric=='noise_auroc' else '1 - cosine similarity')
                        ax.set_ylim((0,1) if metric in ('clean_auroc','noise_auroc') else (0,2)); ax.grid(alpha=.2)
                for k,(protocol,metrics) in enumerate(_raw_baselines(results,model,rows,sigmas).items()):
                    for ax,values in zip(axes,metrics):
                        if any(math.isfinite(value) for value in values):
                            ax.plot(sigmas,values,linestyle='--',marker=MARKERS[k%8],
                                    color=COLORS[(len(methods)+k)%8],label='raw / '+protocol)
                axes[0].set_title(model)
                for ax in axes:
                    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
                save(fig,f'mitigation_fig8_{index}','Input filtering, train augmentation and NCA measurements with recorded raw baselines by protocol. Dashed baselines use robustness AUROC and recorded raw drift; unavailable baselines remain gaps. Source: interventions and robustness.')
            for index,row in enumerate(results.get('token_diagnostics',[]),1):
                maps=[(key,row[key]['drift_map']) for key in ('noise','lesion_occlusion') if isinstance(row.get(key),dict) and 'drift_map' in row[key]]
                if not maps: continue
                fig,axes=plt.subplots(1,len(maps),figsize=(5*len(maps),5),layout='constrained',squeeze=False)
                for ax,(name,values) in zip(axes[0],maps):
                    array=np.asarray(values,dtype=float)
                    if array.ndim==3 and array.shape[0]==1: array=array[0]
                    if array.ndim!=2: raise ValueError('Token maps require one 2D patch grid')
                    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#888888')
                    artist=ax.imshow(np.ma.masked_invalid(array),vmin=0,vmax=2,cmap=cmap,interpolation='nearest')
                    ax.set_title(name); ax.set_xlabel('Patch column'); ax.set_ylabel('Patch row')
                    fig.colorbar(artist,ax=ax,label='1 - cosine similarity')
                save(fig,f'token_drift_{index}',f"Model: {row['model']}; image: {row.get('image_id')}. Shared [0,2] cosine-drift scale; gray means undefined zero-token angle. Spatial evidence only, not causal localization.")
        (staging/'source_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        (staging/'INDEX.md').write_text('# Diagnostic figures\n\nEvidence: '+evidence+'\n\n'
            +'These are diagnostic layouts, not copied paper results. See source_results.json and manifest.json.\n\n'
            +'\n'.join(f"- [{item['name']}]({item['files'][0]}): {item['description']}" for item in manifest['figures'])+'\n')
        _publish_directory(staging,output)
        return manifest
    finally:
        plt.close('all')
        if staging.exists(): shutil.rmtree(staging)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    manifest=build_figures(json.loads(args.results.read_text()),args.output)
    print(json.dumps({'figures':len(manifest['figures']),'evidence':manifest['evidence']}))


if __name__=='__main__': main()
