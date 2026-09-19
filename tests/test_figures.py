import copy
import json
from pathlib import Path
import pytest
from PIL import Image
from report.build_figures import build_figures


def minimal_results():
    return {'metadata':{'evidence':'synthetic_smoke'},'robustness':[
        {'model':'tiny','protocol':'linear','perturbation':'gaussian_noise','severity':.05,'auroc':.6,'drift':.1,'evidence':'synthetic_smoke'},
        {'model':'tiny','protocol':'linear','perturbation':'gaussian_noise','severity':.1,'auroc':None,'drift':None,'evidence':'synthetic_smoke'},
        {'model':'tiny','protocol':'linear','perturbation':'gaussian_noise','severity':.2,'auroc':.5,'drift':.3,'evidence':'synthetic_smoke'}],
        'lesion':[{'model':'tiny','evidence':'synthetic_smoke','delta_lesion':.2,'ci_low':-.1,'ci_high':.4,'n_images':4}],
        'token_diagnostics':[{'model':'tiny','evidence':'synthetic_smoke','image_id':'fixture',
            'noise':{'drift_map':[[[0,None],[.2,.3]]]},'lesion_occlusion':{'drift_map':[[[0,.3],[.4,.5]]]}}]}


def test_figures_keep_source_missing_values_and_export_all_formats(tmp_path):
    source=minimal_results();before=copy.deepcopy(source)
    manifest=build_figures(source,tmp_path/'figures')
    assert source==before and len(manifest['figures'])==3
    assert json.loads((tmp_path/'figures'/'source_results.json').read_text())==source
    for item in manifest['figures']:
        for filename in item['files']:
            assert (tmp_path/'figures'/filename).stat().st_size>100
        with Image.open(tmp_path/'figures'/item['files'][0]) as im:
            assert im.width>=800 and im.height>=600
    with pytest.raises(FileExistsError):build_figures(source,tmp_path/'figures')


def test_invalid_maps_do_not_publish_partial_figures(tmp_path):
    source=minimal_results();source['token_diagnostics'][0]['noise']['drift_map']=[1,2,3]
    with pytest.raises(ValueError,match='2D'):build_figures(source,tmp_path/'figures')
    assert not (tmp_path/'figures').exists()
    source=minimal_results();source['robustness'][0]['evidence']='measured_local'
    with pytest.raises(ValueError,match='Mixed'):build_figures(source,tmp_path/'figures')



def test_fig8_shows_clean_tradeoff_and_keeps_protocols_distinct(tmp_path):
    source=minimal_results();source['robustness']=[];source['lesion']=[];source['token_diagnostics']=[]
    source['interventions']=[{'model':'tiny','protocol':protocol,'intervention':'nca','noise_sigma':.05,
        'clean_auroc':clean,'noise_auroc':.65,'adapted_drift':.1,'evidence':'synthetic_smoke'}
        for protocol,clean in [('linear',.7),('mlp',.8)]]
    build_figures(source,tmp_path/'figures')
    svg=(tmp_path/'figures'/'mitigation_fig8_1.svg').read_text()
    assert 'Clean AUROC' in svg and 'Noisy AUROC' in svg
    assert 'nca / linear' in svg and 'nca / mlp' in svg


def test_token_maps_do_not_silently_discard_a_batch(tmp_path):
    source=minimal_results();source['robustness']=[];source['lesion']=[]
    source['token_diagnostics'][0]['noise']['drift_map']=[[[0,.1],[.2,.3]],[[.4,.5],[.6,.7]]]
    with pytest.raises(ValueError,match='2D'):build_figures(source,tmp_path/'figures')


@pytest.mark.parametrize('section', ['robustness', 'interventions'])
def test_duplicate_plot_conditions_are_rejected_before_publication(tmp_path, section):
    source=minimal_results()
    if section=='robustness':
        duplicate=copy.deepcopy(source['robustness'][0]);duplicate['auroc']=.2
        duplicate['severity']='0.050'
        source['robustness'].append(duplicate)
    else:
        row={'model':'tiny','protocol':'linear','intervention':'blur','noise_sigma':.05,
             'clean_auroc':.8,'noise_auroc':.7,'adapted_drift':.1,'evidence':'synthetic_smoke'}
        source['interventions']=[row,{**row,'noise_sigma':'0.050','noise_auroc':.2}]
    with pytest.raises(ValueError,match='Duplicate'):
        build_figures(source,tmp_path/'figures')
    assert not (tmp_path/'figures').exists()
    assert not list(tmp_path.glob('.figures-*'))


def test_figure_labels_cannot_escape_or_collide_with_output_paths(tmp_path):
    source=minimal_results();source['lesion']=[];source['token_diagnostics']=[]
    original=source['robustness'][0]
    source['robustness']=[{**original,'protocol':protocol,'perturbation':kind}
        for protocol,kind in [('../escaped','gaussian_noise'),
                              (str(tmp_path/'absolute'),'gaussian_noise'),
                              ('a_b','c'),('a','b_c')]]
    sentinels=[tmp_path/'escaped_gaussian_noise_1.png',tmp_path/'absolute_gaussian_noise_1.png']
    for path in sentinels:path.write_bytes(b'KEEP')
    output=tmp_path/'figures';manifest=build_figures(source,output)
    files=[name for item in manifest['figures'] for name in item['files']]
    assert len(manifest['figures'])==4 and len(files)==len(set(files))
    for name in files:
        assert Path(name).name==name and (output/name).is_file()
    assert all(path.read_bytes()==b'KEEP' for path in sentinels)
    assert {path.name for path in tmp_path.iterdir()}=={'figures',*(path.name for path in sentinels)}


@pytest.mark.parametrize('destination_kind',['directory','file','symlink'])
def test_publication_preserves_a_destination_created_by_another_writer(tmp_path,monkeypatch,destination_kind):
    import report.build_figures as module
    publish=module._publish_directory
    output=tmp_path/'figures';observed={}
    target=tmp_path/'other';target.mkdir();(target/'sentinel').write_bytes(b'KEEP')
    def compete(staging,destination):
        if destination_kind=='directory':destination.mkdir()
        elif destination_kind=='file':destination.write_bytes(b'KEEP')
        else:destination.symlink_to(target,target_is_directory=True)
        observed['inode']=destination.lstat().st_ino
        publish(staging,destination)
    monkeypatch.setattr(module,'_publish_directory',compete)
    source={'metadata':{'evidence':'synthetic_smoke'}}
    with pytest.raises(FileExistsError):build_figures(source,output)
    assert output.lstat().st_ino==observed['inode']
    if destination_kind=='file':assert output.read_bytes()==b'KEEP'
    elif destination_kind=='directory':assert list(output.iterdir())==[]
    else:assert output.is_symlink() and (output/'sentinel').read_bytes()==b'KEEP'
    assert not list(tmp_path.glob('.figures-*'))


def _capture_fig8(monkeypatch):
    from matplotlib.figure import Figure
    captured=[]
    def save(self,path,*args,**kwargs):
        if Path(path).name=='mitigation_fig8_1.png':
            captured.extend([{line.get_label():list(line.get_ydata()) for line in axis.lines}
                             for axis in self.axes])
        Path(path).write_bytes(b'figure stub')
    monkeypatch.setattr(Figure,'savefig',save)
    return captured


def test_fig8_preserves_raw_clean_noisy_drift_baselines_by_protocol(tmp_path,monkeypatch):
    captured=_capture_fig8(monkeypatch)
    evidence='synthetic_smoke'
    source={'metadata':{'evidence':evidence},'robustness':[], 'interventions':[]}
    for protocol,clean,noisy,drift in [('linear',.91,.2,.8),('mlp',.82,.3,.6)]:
        base={'model':'tiny','protocol':protocol,'evidence':evidence}
        source['robustness'].extend([{**base,'perturbation':'clean','severity':0,'auroc':clean,'drift':0},
            {**base,'perturbation':'gaussian_noise','severity':.05,'auroc':noisy,'drift':drift}])
    for protocol in ('linear','linear_train_aug','nca','mlp'):
        source['interventions'].append({'model':'tiny','protocol':protocol,'intervention':'adapted',
            'noise_sigma':.05,'clean_auroc':.7,'noise_auroc':.5,'adapted_drift':.1,
            'raw_drift':.6 if protocol=='mlp' else .8,'evidence':evidence})
    build_figures(source,tmp_path/'figures')
    assert [axis['raw / linear'] for axis in captured]==[[.91],[.2],[.8]]
    assert [axis['raw / mlp'] for axis in captured]==[[.82],[.3],[.6]]
    assert all('raw / nca' not in axis and 'raw / linear_train_aug' not in axis for axis in captured)
    assert captured[0]['adapted / nca']==[.7]
    assert captured[1]['adapted / linear_train_aug']==[.5]
    assert captured[2]['adapted / mlp']==[.1]


def test_fig8_uses_recorded_raw_drift_without_inventing_missing_accuracy(tmp_path,monkeypatch):
    import math
    captured=_capture_fig8(monkeypatch)
    source={'metadata':{'evidence':'synthetic_smoke'},'interventions':[
        {'model':'tiny','protocol':'nca','intervention':'adapted','noise_sigma':sigma,
         'clean_auroc':.7,'noise_auroc':.5,'adapted_drift':.1,'raw_drift':raw,'evidence':'synthetic_smoke'}
        for sigma,raw in [(.05,.8),(.1,None)]]}
    build_figures(source,tmp_path/'figures')
    assert 'raw / linear' not in captured[0] and 'raw / linear' not in captured[1]
    assert captured[2]['raw / linear'][0]==.8
    assert math.isnan(captured[2]['raw / linear'][1])


def test_conflicting_raw_baselines_do_not_publish_a_figure(tmp_path,monkeypatch):
    _capture_fig8(monkeypatch)
    source={'metadata':{'evidence':'synthetic_smoke'},'interventions':[
        {'model':'tiny','protocol':protocol,'intervention':'adapted','noise_sigma':.05,
         'clean_auroc':.7,'noise_auroc':.5,'adapted_drift':.1,'raw_drift':raw,'evidence':'synthetic_smoke'}
        for protocol,raw in [('linear',.8),('nca',.7)]]}
    with pytest.raises(ValueError,match='Conflicting recorded raw drift'):
        build_figures(source,tmp_path/'figures')
    assert not (tmp_path/'figures').exists()
    assert not list(tmp_path.glob('.figures-*'))



@pytest.mark.parametrize('relative_output',['figures','nested/figures','nested/../figures'])
def test_figures_accept_relative_output_paths(tmp_path,monkeypatch,relative_output):
    monkeypatch.chdir(tmp_path)
    source={'metadata':{'evidence':'synthetic_smoke'}}
    manifest=build_figures(source,relative_output)
    output=Path(relative_output).absolute()
    # A lexical .. path may not have its unused intermediate directory created.
    import os
    output=Path(os.path.abspath(output))
    assert manifest['figures']==[]
    assert json.loads((output/'source_results.json').read_text())==source
    assert not list(output.parent.glob('.figures-*'))


def test_relative_output_still_rejects_symlink_parents(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'target').mkdir()
    (tmp_path/'alias').symlink_to(tmp_path/'target',target_is_directory=True)
    source={'metadata':{'evidence':'synthetic_smoke'}}
    with pytest.raises(ValueError,match='symlinks'):
        build_figures(source,'alias/../figures')
    assert not (tmp_path/'figures').exists()
