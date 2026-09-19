"""Standalone baseline entrypoint stays plan-only unless execution is explicit."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch

from scripts.pretrain import main, load_pretrain_config, execute_pretraining, normalize_pretrain_config


def configuration(tmp_path, architecture):
    root = tmp_path / "images"
    root.mkdir()
    samples = []
    for index in range(4):
        path = root / f"image-{index}.png"
        Image.fromarray(np.full((32, 32, 3), index * 50, dtype=np.uint8)).save(path)
        samples.append({"image_id": str(index), "image": path.name})
    (tmp_path / "images.json").write_text(json.dumps({"samples": samples}))
    raw = {"schema_version": 1, "architecture": architecture,
           "runtime": {"profile": "fixture", "device": "cpu", "batch_size": 2,
                       "output_dir": "output", "seed": 7},
           "data": {"unlabeled_manifest": "images.json", "image_root": "images"},
           "model": {"embed_dim": 8, "num_heads": 2, "predictor_dim": 8, "predictor_depth": 1},
           "normalization": {"mean": [.4, .4, .4], "std": [.3, .3, .3]},
           "training": {"epochs": 2}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path, raw


def test_default_plan_does_not_decode_images_or_build_model(tmp_path, monkeypatch, capsys):
    path, raw = configuration(tmp_path, "ijepa")
    def forbidden(*args, **kwargs):
        raise AssertionError("default plan performed image/model work")
    monkeypatch.setattr("scripts.pretrain.build_full_ijepa", forbidden)
    monkeypatch.setattr("scripts.pretrain.MAE", forbidden)
    monkeypatch.setattr("PIL.Image.open", forbidden)
    result = main(["--config", str(path)])
    assert result["action"] == "plan" and result["unlabeled_samples"] == 4
    assert result["model_constructed"] is False and result["images_decoded"] is False
    assert not (tmp_path / "output").exists()
    assert json.loads(capsys.readouterr().out)["requires_explicit_execute"] is True


@pytest.mark.parametrize("architecture", ["ijepa", "mae"])
def test_explicit_execute_and_resume_preserve_actual_epochs(tmp_path, architecture):
    path, raw = configuration(tmp_path, architecture)
    config = load_pretrain_config(path)
    result = execute_pretraining(config)
    assert result["action"] == "completed" and result["evidence"] == "synthetic_smoke"
    assert result["training"]["completed_epochs"] == 2
    assert result["training"]["actual_training_steps"] == 4
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert payload["metadata"]["epoch"] == 2
    assert payload["metadata"]["epoch_is_lineage_label"] is False
    assert payload["metadata"]["mock"] is True
    assert payload["metadata"]["variant"] == "baseline_" + architecture
    resumed = execute_pretraining(config, resume=True)
    assert resumed["training"]["history"] == result["training"]["history"]
    with pytest.raises(FileExistsError, match="resume"):
        execute_pretraining(config)


def test_resume_after_explicit_epoch_boundary_stop(tmp_path):
    path, raw = configuration(tmp_path, "mae")
    raw["training"]["max_steps"] = 2
    config = normalize_pretrain_config(raw, base=tmp_path)
    partial = execute_pretraining(config)
    assert partial["action"] == "stopped_at_epoch_boundary"
    assert partial["training"]["completed_epochs"] == 1
    raw["training"]["max_steps"] = None
    resumed = execute_pretraining(normalize_pretrain_config(raw, base=tmp_path), resume=True)
    assert resumed["action"] == "completed"
    assert resumed["training"]["actual_training_steps"] == 4


def test_fixture_bounds_and_resume_acknowledgement(tmp_path):
    path, raw = configuration(tmp_path, "ijepa")
    raw["model"]["embed_dim"] = 1280
    with pytest.raises(ValueError, match="tiny CPU"):
        normalize_pretrain_config(raw, base=tmp_path)
    with pytest.raises(SystemExit):
        main(["--config", str(path), "--resume"])


def test_direct_execution_revalidates_fixture_bounds(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path, "ijepa")
    config = load_pretrain_config(path)
    config["model"]["embed_dim"] = 1280
    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized fixture reached model construction")
    monkeypatch.setattr("scripts.pretrain.build_full_ijepa", forbidden)
    with pytest.raises(ValueError, match="tiny CPU"):
        execute_pretraining(config)


@pytest.mark.parametrize('filename',['epoch_0001.pt','pretraining_result.json'])
def test_new_run_protects_orphan_outputs(tmp_path,filename):
    path,_=configuration(tmp_path,'ijepa'); config=load_pretrain_config(path)
    output=Path(config['runtime']['output_dir']);output.mkdir()
    existing=output/filename;existing.write_bytes(b'keep existing artifact')
    with pytest.raises(FileExistsError,match='nonempty'):execute_pretraining(config)
    assert existing.read_bytes()==b'keep existing artifact'


def test_resume_recovers_a_complete_epoch_without_latest_pointer(tmp_path):
    path,_=configuration(tmp_path,'ijepa');config=load_pretrain_config(path)
    original=execute_pretraining(config)
    latest=Path(original['checkpoint']);latest.unlink()
    resumed=execute_pretraining(config,resume=True)
    assert resumed['training']['history']==original['training']['history']
    assert latest.is_file()



def test_output_symlink_is_not_erased_by_config_normalization(tmp_path):
    path,_=configuration(tmp_path,'ijepa');target=tmp_path/'other';target.mkdir()
    (tmp_path/'output').symlink_to(target,target_is_directory=True)
    config=load_pretrain_config(path)
    assert Path(config['runtime']['output_dir']).is_symlink()
    with pytest.raises(ValueError,match='symlinks'): execute_pretraining(config)
    assert not list(target.iterdir())
