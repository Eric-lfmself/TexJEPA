"""Offline fixture jobs exercise strict native lineage without GPU or downloads."""
import copy
import json

import numpy as np
import pytest
from PIL import Image
import torch

from configs import load_config
from models import save_checkpoint
from scripts.post_training_jobs import build_full_ijepa, execute_post_training


def fixture_config(tmp_path, variants=None):
    root = tmp_path / "images"
    root.mkdir()
    samples = []
    for index in range(4):
        path = root / f"unlabeled-{index}.png"
        array = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(array).save(path)
        samples.append({"image_id": path.stem, "image": path.name})
    manifest = tmp_path / "unlabeled.json"
    manifest.write_text(json.dumps({"samples": samples}))
    paper = load_config("smoke")
    model_spec = {**paper["model"], "embed_dim": 8, "num_heads": 2, "predictor_dim": 8, "predictor_depth": 1}
    model = build_full_ijepa(model_spec)
    baseline = save_checkpoint(tmp_path / "v31.pt", model, {"variant": "v3.1", "epoch": 201,
                                "evidence": "synthetic_smoke", "mock": True, "epoch_is_lineage_label": True})
    v4_parent = save_checkpoint(tmp_path / "v4-parent.pt", model, {"variant": "v4", "epoch": 50,
                                "evidence": "synthetic_smoke", "mock": True, "epoch_is_lineage_label": True})
    return {"runtime": {"profile": "fixture", "device": "cpu", "batch_size": 2,
                        "num_workers": 0, "seed": 42},
            "data": {"unlabeled_manifest": str(manifest), "unlabeled_root": str(root), "image_size": 32,
                     "intensity": {"policy": "uint8", "scale": None}},
            "post_training": {"enabled": True, "variants": variants or ["v4", "v5", "v6"],
                "baseline_checkpoint": str(baseline), "parent_checkpoints": {"v5": str(v4_parent), "v6": str(v4_parent)},
                "backbone": model_spec, "normalization": {"mean": [.5, .5, .5], "std": [.25, .25, .25]},
                "epochs": {"v4": 2, "v5": 2, "v6": 2}, "training": {"learning_rate": .001},
                "objective": paper["post_training"]}}


def test_full_native_jobs_preserve_fixture_evidence_and_support_eval_only_resume(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    records = execute_post_training(config, tmp_path / "jobs")
    assert [record["variant"] for record in records] == ["v4", "v5", "v6"]
    for record in records:
        assert record["metadata"]["mock"] is True
        assert record["metadata"]["evidence"] == "synthetic_smoke"
        assert record["metadata"]["epoch"] == 2
        assert record["metadata"]["actual_training_steps"] == 4
        assert record["metadata"]["epoch_is_lineage_label"] is False
        assert record["training"]["completed_epochs"] == 2
        assert record["metadata"]["normalization"] == config["post_training"]["normalization"]
        assert record["parent"]["warm_start"]["epoch_is_lineage_label"] is True
    assert not (tmp_path / "jobs" / "v4" / "epoch_0050.pt").exists()
    resumed = execute_post_training(config, tmp_path / "jobs", resume=True)
    assert [record["training"]["history"] for record in records] == [record["training"]["history"] for record in resumed]
    def forbidden(*args, **kwargs):
        raise AssertionError("eval-only attempted fitting")
    monkeypatch.setattr("scripts.post_training_jobs.train_post_training", forbidden)
    evaluated = execute_post_training(config, tmp_path / "jobs", eval_only=True)
    assert all(record["training"]["evaluation_only"] for record in evaluated)
    assert all(record["checkpoint_sha256"] for record in evaluated)


def test_real_mode_rejects_synthetic_parent_before_training(tmp_path, monkeypatch):
    config = fixture_config(tmp_path, ["v4"])
    config["runtime"]["profile"] = "experiment"
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid parent caused model allocation")
    monkeypatch.setattr("scripts.post_training_jobs.build_full_ijepa", forbidden)
    with pytest.raises(ValueError, match="mock, synthetic or fixture"):
        execute_post_training(config, tmp_path / "jobs")
    assert not (tmp_path / "jobs").exists()


def test_fixture_workload_and_missing_epoch50_rejected(tmp_path):
    config = fixture_config(tmp_path, ["v5"])
    config["post_training"]["parent_checkpoints"] = {}
    with pytest.raises(FileNotFoundError, match="v5"):
        execute_post_training(config, tmp_path / "jobs")
    config["post_training"]["epochs"]["v5"] = 50
    with pytest.raises(ValueError, match="at most two epochs"):
        execute_post_training(config, tmp_path / "jobs")


def test_changed_corpus_blocks_eval_only_and_existing_runs_need_resume(tmp_path):
    config = fixture_config(tmp_path, ["v4"])
    execute_post_training(config, tmp_path / "jobs")
    with pytest.raises(FileExistsError, match="resume"):
        execute_post_training(config, tmp_path / "jobs")
    manifest = config["data"]["unlabeled_manifest"]
    from pathlib import Path
    raw = json.loads(Path(manifest).read_text())
    raw["samples"].reverse()
    Path(manifest).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="configured post-training job"):
        execute_post_training(config, tmp_path / "jobs", eval_only=True)


@pytest.mark.parametrize("ancestry", [
    {"parent": {"warm_start": {"mock": True, "evidence": "synthetic_smoke"}}},
    {"warm_start": {"parent": {"epoch_is_lineage_label": True}}},
    {"parents": [{"source": "explicit_random_initialization"}]},
])
def test_real_mode_rejects_laundered_fixture_ancestry(tmp_path, ancestry):
    config = fixture_config(tmp_path, ["v4"])
    config["runtime"]["profile"] = "experiment"
    path = config["post_training"]["baseline_checkpoint"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"] = {"variant": "v3.1", "epoch": 201, "mock": False,
                           "epoch_is_lineage_label": False, "evidence": "measured_local", **ancestry}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="mock, synthetic or fixture"):
        execute_post_training(config, tmp_path / "jobs")


@pytest.mark.parametrize('evidence',['synthetic','fixture'])
def test_real_parent_aliases_are_rejected_recursively(tmp_path,evidence,monkeypatch):
    from scripts.post_training_jobs import _evidence_guard
    metadata={'mock':False,'epoch_is_lineage_label':False,'evidence':'measured_local',
              'parent':{'warm_start':{'evidence':evidence}}}
    with pytest.raises(ValueError,match='mock, synthetic'):_evidence_guard(metadata,'experiment','warm start')


def test_orphan_post_epoch_is_preserved_and_valid_resume_can_recover(tmp_path):
    config=fixture_config(tmp_path,['v4']);folder=tmp_path/'jobs'
    records=execute_post_training(config,folder)
    latest=folder/'v4'/'latest.pt'; latest.unlink()
    epoch=folder/'v4'/'epoch_0002.pt';before=epoch.read_bytes()
    with pytest.raises(FileExistsError,match='nonempty'):execute_post_training(config,folder)
    assert epoch.read_bytes()==before
    resumed=execute_post_training(config,folder,resume=True)
    assert latest.is_file()
    assert resumed[0]['training']['history']==records[0]['training']['history']
