"""Execution guard tests: paper defaults must never allocate a Huge model."""
import pytest
from configs import load_config, validate_config


def test_paper_is_description_only():
    cfg = load_config("paper")
    assert (cfg["model"]["embed_dim"], cfg["model"]["predictor_depth"]) == (1280, 12)
    with pytest.raises(ValueError, match="synthetic"):
        validate_config(cfg, for_execution=True)


def test_cpu_smoke_and_unknown_keys():
    validate_config(load_config(), for_execution=True)
    with pytest.raises(ValueError, match="CPU"):
        load_config(overrides={"runtime": {"device": "cuda"}})
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides={"runtime": {"download": True}})
    with pytest.raises(ValueError, match="oversized"):
        validate_config(load_config(overrides={"model": {"depth": 32}}), for_execution=True)


@pytest.mark.parametrize("override", [{"predictor_dim":8192},{"num_register_tokens":100},
                                      {"in_channels":100},{"patch_size":1}])
def test_allocation_knobs_are_bounded(override):
    with pytest.raises(ValueError,match="oversized"):
        validate_config(load_config(overrides={"model":override}), for_execution=True)


@pytest.mark.parametrize("override", [{"depth":1},{"patch_size":32},{"in_channels":1}])
def test_whole_pipeline_compatibility(override):
    with pytest.raises(ValueError):
        validate_config(load_config(overrides={"model":override}), for_execution=True)
