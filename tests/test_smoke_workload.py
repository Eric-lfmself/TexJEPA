"""Reject unsafe smoke work before allocation; retain precise frequency sweeps."""
import csv
import math

import pytest

from configs import load_config, validate_config
from scripts import run_all


@pytest.mark.parametrize("section,key,limit", [
    ("lesion", "controls", 5),
    ("lesion", "bootstrap_replicates", 2000),
    ("post_training", "target_blocks", 6),
    ("post_training", "register_target_blocks", 6),
])
def test_work_multipliers_are_bounded_before_models_or_outputs(tmp_path, monkeypatch, section, key, limit):
    config = load_config(overrides={section: {key: limit + 1}})
    monkeypatch.setattr(run_all, "MiniViT", lambda **kwargs: pytest.fail("model allocated before validation"))
    destination = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Smoke"):
        run_all.run_pipeline(config, destination)
    assert not destination.exists()


def test_workload_boundaries_and_smaller_settings_remain_supported():
    for controls, bootstrap, targets in ((1, 1, 1), (5, 2000, 6)):
        config = load_config(overrides={
            "lesion": {"controls": controls, "bootstrap_replicates": bootstrap},
            "post_training": {"target_blocks": targets, "register_target_blocks": targets},
            "perturbations": {"gaussian_blur": [0, 4]},
        })
        validate_config(config, for_execution=True)


@pytest.mark.parametrize("section,key,value", [
    ("runtime", "batch_size", 2.5),
    ("runtime", "num_threads", True),
    ("data", "smoke_samples", 4.5),
    ("probing", "hidden_dim", math.inf),
    ("probing", "smoke_epochs", True),
    ("lesion", "controls", False),
    ("lesion", "controls", 0),
    ("lesion", "bootstrap_replicates", 200.0),
    ("post_training", "target_blocks", -1),
    ("post_training", "register_target_blocks", math.nan),
    ("nca", "hidden_dim", 32.0),
])
def test_counts_are_positive_integers(section, key, value):
    with pytest.raises(ValueError, match="positive integer"):
        load_config(overrides={section: {key: value}})


@pytest.mark.parametrize("override", [
    {"perturbations": {"gaussian_noise": [math.nan]}},
    {"perturbations": {"brightness_shift": [math.inf]}},
    {"perturbations": {"contrast_scaling": [-1]}},
    {"probing": {"head_lr": math.nan}},
    {"nca": {"feature_weight": math.inf}},
    {"post_training": {"poisson_rate": 0}},
    {"post_training": {"jpeg_quality": 101}},
    {"post_training": {"context_scale": [0, 1]}},
    {"post_training": {"register_context_scale": [.85, .65]}},
    {"post_training": {"target_scale": [.1, 1.1]}},
    {"post_training": {"target_scale": [.1, math.nan]}},
    {"post_training": {"target_aspect": [0, 2]}},
    {"perturbations": {"frequency_bands": [[.2, .2]]}},
    {"perturbations": {"frequency_bands": [[0, 1.1]]}},
    {"perturbations": {"low_pass_cutoff": 0}},
])
def test_invalid_numeric_domains_fail_early(override):
    with pytest.raises(ValueError):
        load_config(overrides=override)


def test_blur_kernel_growth_is_rejected_before_execution(tmp_path):
    for sigma in (4.01, 10000.0):
        config = load_config(overrides={"perturbations": {"gaussian_blur": [sigma]}})
        with pytest.raises(ValueError, match="blur sigma"):
            run_all.run_pipeline(config, tmp_path / "unsafe_blur")
    assert not (tmp_path / "unsafe_blur").exists()


def test_total_corruption_count_includes_frequency_mode_product():
    config = load_config()
    # Other default lists contribute 18 conditions, including all six band/mode pairs.
    config["perturbations"]["gaussian_noise"] = [i / 100 for i in range(14)]
    assert len(run_all.configured_perturbations(config)) == 32
    validate_config(config, for_execution=True)
    config["perturbations"]["gaussian_noise"].append(.99)
    with pytest.raises(ValueError, match="32 perturbation"):
        validate_config(config, for_execution=True)


def test_disabled_band_modes_cannot_hide_an_oversized_band_list():
    config = load_config(overrides={"perturbations": {
        "band_modes": [], "frequency_bands": [[i / 1000, (i + 1) / 1000] for i in range(33)]}})
    with pytest.raises(ValueError, match="entries per list"):
        validate_config(config, for_execution=True)


@pytest.mark.parametrize("override", [
    {"gaussian_noise": [0.0, -0.0]},
    {"gaussian_blur": [1, 1.0]},
    {"brightness_shift": [.1, .1]},
    {"contrast_scaling": [1, 1.0]},
    {"frequency_bands": [[0, .2], [-0.0, .20]]},
    {"band_modes": ["stop", "stop"]},
])
def test_numeric_duplicates_fail_before_fitting(tmp_path, monkeypatch, override):
    config = load_config()
    config["perturbations"].update(override)
    monkeypatch.setattr(run_all, "fit_probe", lambda *args, **kwargs: pytest.fail("fitting preceded duplicate check"))
    with pytest.raises(ValueError, match="Duplicate"):
        run_all.run_pipeline(config, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_default_band_labels_are_unchanged():
    labels = [spec.severity for spec in run_all.configured_perturbations(load_config())
              if spec.name == "band_stop"]
    assert labels == ["0.00-0.20", "0.20-0.45", "0.45-1.00"]
    assert run_all._band_endpoint(-0.0) == "0.00"


def test_close_band_endpoints_survive_full_pipeline_and_table(tmp_path):
    config = load_config(overrides={"perturbations": {"frequency_bands": [[.001, .201], [.002, .202]]}})
    result = run_all.run_pipeline(config, tmp_path)
    bands = [row for row in result["robustness"] if row["perturbation"] == "band_stop"]
    assert {row["severity"] for row in bands} == {"0.001-0.201", "0.002-0.202"}
    assert all(row["evidence"] == "synthetic_smoke" for row in bands)
    with (tmp_path / "tables" / "table_VI.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["Condition"] == "band_stop"]
    assert {row["Severity"] for row in rows} == {"0.001-0.201", "0.002-0.202"}
    assert result["metadata"]["real_experiments_run"] is False
    assert not set(result["metadata"]["train_image_ids"]) & set(result["metadata"]["test_image_ids"])
