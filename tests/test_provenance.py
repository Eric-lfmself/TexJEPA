import pytest
from models.provenance import SYNTHETIC_SOURCES
from scripts.run_experiments import _reject_synthetic,_checkpoint_metadata_guard
from scripts.post_training_jobs import _evidence_guard
import torch


@pytest.mark.parametrize('source',SYNTHETIC_SOURCES)
def test_known_synthetic_source_markers_match_across_real_entrypoints(tmp_path,source):
    metadata={'mock':False,'epoch_is_lineage_label':False,'evidence':'measured_local',
              'parent':{'source':source}}
    with pytest.raises(ValueError):_reject_synthetic(metadata,'local')
    with pytest.raises(ValueError):_evidence_guard(metadata,'experiment','warm start')
    path=tmp_path/'weights.pt';torch.save({'model':{'x':torch.ones(1)},'metadata':metadata},path)
    with pytest.raises(ValueError):_checkpoint_metadata_guard({'checkpoint':{'path':str(path)}},True)
