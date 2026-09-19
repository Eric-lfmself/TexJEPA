"""Whole-run publication preserves prior evidence and caller-owned files."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from configs import load_config
from scripts import run_all


def tiny_config(seed=42):
    return load_config(overrides={
        "runtime": {"seed": seed, "device": "cpu", "batch_size": 2, "num_threads": 1},
        "data": {"smoke_samples": 4},
        "perturbations": {"gaussian_noise": [.05], "gaussian_blur": [],
                          "brightness_shift": [], "contrast_scaling": [],
                          "frequency_bands": [], "band_modes": []},
    })


def snapshot(folder):
    return {str(path.relative_to(folder)): ("link", str(path.readlink())) if path.is_symlink()
            else ("file", path.read_bytes())
            for path in folder.rglob("*") if path.is_symlink() or path.is_file()}


def fail(*args, **kwargs):
    raise RuntimeError("injected stage failure")


def assert_no_staging(folder):
    assert not list(folder.parent.glob(f".{folder.name}.staging-*"))
    assert not list(folder.parent.glob(f".{folder.name}.previous-*"))


@pytest.mark.parametrize("failure_stage", ["fit_probe", "build_tables"])
def test_failed_rerun_preserves_all_old_artifacts(tmp_path, monkeypatch, failure_stage):
    output = tmp_path / "run"
    run_all.run_pipeline(tiny_config(), output)
    before = snapshot(output)
    monkeypatch.setattr(run_all, failure_stage, fail)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        run_all.run_pipeline(tiny_config(43), output)
    assert snapshot(output) == before
    assert_no_staging(output)


def test_failed_first_run_has_no_published_directory(tmp_path, monkeypatch):
    output = tmp_path / "new_run"
    monkeypatch.setattr(run_all, "fit_probe", fail)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        run_all.run_pipeline(tiny_config(), output)
    assert not output.exists()
    assert_no_staging(output)


def add_extras(output, outside):
    output.mkdir(exist_ok=True)
    (output / "notes").mkdir()
    (output / "notes" / "keep.txt").write_text("caller notes")
    for directory in ("checkpoints", "tables"):
        (output / directory).mkdir(exist_ok=True)
        (output / directory / "caller.txt").write_text(f"keep {directory}")
    outside.write_text("external symlink target")
    (output / "notes.link").symlink_to(outside)
    return (output / "notes").stat().st_ino


def test_publication_failure_restores_old_run_and_unknown_entries(tmp_path, monkeypatch):
    output = tmp_path / "run"
    run_all.run_pipeline(tiny_config(), output)
    add_extras(output, tmp_path / "outside.txt")
    before = snapshot(output)
    replace = run_all.os.replace

    def refuse_final_rename(source, target):
        if Path(source).name.startswith(".run.staging-") and Path(target) == output:
            raise OSError("injected publication failure")
        return replace(source, target)

    monkeypatch.setattr(run_all.os, "replace", refuse_final_rename)
    with pytest.raises(OSError, match="publication failure"):
        run_all.run_pipeline(tiny_config(43), output)
    assert snapshot(output) == before
    assert (tmp_path / "outside.txt").read_text() == "external symlink target"
    assert_no_staging(output)


def assert_lineage_paths(value, output):
    if isinstance(value, dict):
        if "source_path" in value:
            path = Path(value["source_path"])
            assert path.is_relative_to(output)
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == value["source_sha256"]
        for item in value.values():
            assert_lineage_paths(item, output)
    elif isinstance(value, list):
        for item in value:
            assert_lineage_paths(item, output)


def test_success_preserves_unknown_files_and_publishes_resolvable_lineage(tmp_path):
    output = tmp_path / "run"
    notes_inode = add_extras(output, tmp_path / "outside.txt")
    result = run_all.run_pipeline(tiny_config(), output)
    assert (output / "notes").stat().st_ino == notes_inode
    assert (output / "notes" / "keep.txt").read_text() == "caller notes"
    assert (output / "checkpoints" / "caller.txt").read_text() == "keep checkpoints"
    assert (output / "tables" / "caller.txt").read_text() == "keep tables"
    assert (output / "notes.link").is_symlink()
    assert (tmp_path / "outside.txt").read_text() == "external symlink target"
    assert json.loads((output / "results.json").read_text()) == result
    assert ".staging-" not in json.dumps(result)
    assert_lineage_paths(result, output)
    for checkpoint in result["metadata"]["synthetic_checkpoints"]:
        path = Path(checkpoint["path"])
        assert path.is_relative_to(output)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert ".staging-" not in json.dumps(payload["metadata"])
        assert_lineage_paths(payload["metadata"], output)
    assert_no_staging(output)


@pytest.mark.parametrize("position", ["tables", "managed_file"])
def test_owned_symlink_outputs_are_rejected_before_fitting(tmp_path, monkeypatch, position):
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    if position == "tables":
        (output / "tables").symlink_to(outside, target_is_directory=True)
    else:
        (outside / "result.json").write_text("caller data")
        (output / "results.json").symlink_to(outside / "result.json")
    monkeypatch.setattr(run_all, "MiniViT", fail)
    with pytest.raises(ValueError, match="symlink"):
        run_all.run_pipeline(tiny_config(), output)
    assert outside.is_dir()
    assert_no_staging(output)


def test_unrecognized_generated_name_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "results.json").write_text('{"my_data": true}')
    before = snapshot(output)
    monkeypatch.setattr(run_all, "MiniViT", fail)
    with pytest.raises(ValueError, match="valid prior smoke result"):
        run_all.run_pipeline(tiny_config(), output)
    assert snapshot(output) == before
    assert_no_staging(output)


def test_zero_nca_objective_rejected_before_any_output_or_model(tmp_path, monkeypatch):
    config = tiny_config()
    config["nca"].update(feature_weight=0., prediction_weight=0., supervised_weight=0.)
    output = tmp_path / "zero"
    monkeypatch.setattr(run_all, "MiniViT", fail)
    with pytest.raises(ValueError, match="NCA loss weight"):
        run_all.run_pipeline(config, output)
    assert not output.exists()
    assert_no_staging(output)


def test_pipeline_reuses_equal_unmodified_rows_without_mutating_them(tmp_path, monkeypatch):
    evaluate = run_all.evaluate_mitigations
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(kwargs["include_unmodified"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(run_all, "evaluate_mitigations", wrapped)
    result = run_all.run_pipeline(tiny_config(), tmp_path / "run")
    assert calls == [False, False]
    for name in {row["model"] for row in result["mitigation"]}:
        original = [{key: value for key, value in row.items() if key not in ("group", "paper_model")}
                    for row in result["robustness"] if row["model"] == name and row["protocol"] == "linear"]
        reused = [{key: value for key, value in row.items() if key not in ("mitigation", "mitigation_parameters")}
                  for row in result["mitigation"] if row["model"] == name and row["mitigation"] == "none"]
        assert reused == original
    assert all("mitigation" not in row for row in result["robustness"])


def test_output_directory_alias_is_preserved_and_updates_its_target(tmp_path):
    target = tmp_path / "real_output"
    target.mkdir()
    (target / "caller.txt").write_text("keep alias target notes")
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    result = run_all.run_pipeline(tiny_config(), alias)
    assert alias.is_symlink()
    assert alias.resolve() == target
    assert (target / "caller.txt").read_text() == "keep alias target notes"
    assert json.loads((alias / "results.json").read_text()) == result
    assert_lineage_paths(result, target)
    assert_no_staging(target)
