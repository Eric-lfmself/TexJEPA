"""Optional offline I-JEPA/MAE baseline training with plan as the default action.

Usage: python -m scripts.pretrain --config job.json [--execute] [--resume]
This is separate from paper diagnostics, which normally consume author weights.
No datasets, architectures or weights are downloaded. Native architecture and
unreported optimization settings are explicit implementation choices.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs import load_config
from data import UnlabeledManifestDataset
from data.experiment import inspect_manifest
from models import MAE, MiniViT, BlockMaskSampler
from models.checkpoints import checkpoint_sha256
from scripts.post_training_jobs import build_full_ijepa, ImageNormalizer
from training import TrainingConfig, train_pretraining
from training.state import seed_runtime


def _merge(defaults, updates, name):
    if not isinstance(updates, dict) or set(updates) - set(defaults):
        raise ValueError(f"Unknown {name} configuration fields")
    return {**defaults, **updates}


def _path(value, base, name):
    if not isinstance(value, str) or not value.strip() or "://" in value or "\x00" in value:
        raise ValueError(f"{name} must be an explicit local path")
    path=base / Path(value).expanduser()
    return os.path.abspath(path) if name=='runtime.output_dir' else str(path.resolve())


def normalize_pretrain_config(raw, *, base=None):
    """Validate dimensions and settings without constructing any model."""
    base = Path(base or Path.cwd())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Baseline training requires schema_version=1")
    allowed = {"schema_version", "architecture", "variant", "runtime", "data", "model",
               "normalization", "training", "masking", "mae"}
    if set(raw) - allowed:
        raise ValueError("Unknown baseline training configuration sections")
    architecture = raw.get("architecture", "ijepa")
    if architecture not in ("ijepa", "mae"):
        raise ValueError("architecture must be ijepa or mae")
    runtime = _merge({"profile": "experiment", "device": "cuda:0", "seed": 42,
                       "batch_size": 32, "num_workers": 0, "num_threads": 1,
                       "output_dir": "outputs/pretraining"}, raw.get("runtime", {}), "runtime")
    if runtime["profile"] not in ("experiment", "fixture"):
        raise ValueError("Use an explicit experiment or fixture profile")
    for key in ("seed", "batch_size", "num_workers", "num_threads"):
        if type(runtime[key]) is not int or runtime[key] < (1 if key in ("batch_size", "num_threads") else 0):
            raise ValueError(f"runtime.{key} must be a valid integer")
    runtime["output_dir"] = _path(runtime["output_dir"], base, "runtime.output_dir")
    paper = load_config("smoke" if runtime["profile"] == "fixture" else "paper")
    model = _merge(paper["model"], raw.get("model", {}), "model")
    for key in ("image_size", "patch_size", "in_channels", "embed_dim", "depth", "num_heads", "predictor_dim", "predictor_depth"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError("Model dimensions must be positive integers")
    if model["image_size"] % model["patch_size"] or model["embed_dim"] % model["num_heads"] or model["predictor_dim"] % model["num_heads"]:
        raise ValueError("Image/patch and attention dimensions must divide exactly")
    if model["num_register_tokens"] != 0:
        raise ValueError("Baseline pretraining uses no register tokens")
    data = _merge({"unlabeled_manifest": None, "image_root": None,
                   "intensity": {"policy": "uint8", "scale": None}}, raw.get("data", {}), "data")
    for key in ("unlabeled_manifest", "image_root"):
        data[key] = _path(data[key], base, "data." + key)
    data["intensity"] = _merge({"policy": "uint8", "scale": None}, data["intensity"], "intensity")
    intensity = data["intensity"]
    if intensity["policy"] == "uint8":
        if intensity["scale"] is not None:
            raise ValueError("uint8 intensity does not accept a scale")
    elif intensity["policy"] == "uint16_scale":
        value = intensity["scale"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 65535:
            raise ValueError("uint16_scale requires an explicit scale in (0,65535]")
    else:
        raise ValueError("Unknown intensity policy")
    normalization = raw.get("normalization", {"mean": [0.] * model["in_channels"], "std": [1.] * model["in_channels"]})
    if not isinstance(normalization, dict) or set(normalization) != {"mean", "std"}:
        raise ValueError("normalization must contain channel mean/std")
    for key in ("mean", "std"):
        values = normalization[key]
        if (not isinstance(values, list) or len(values) != model["in_channels"]
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or key == "std" and any(v <= 0 for v in values)):
            raise ValueError("Invalid normalization channel values")
    training_input = raw.get("training", {})
    if not isinstance(training_input, dict) or set(training_input) - {item.name for item in fields(TrainingConfig)}:
        raise ValueError("Unknown TrainingConfig settings")
    if any(key in training_input and training_input[key] != runtime[key] for key in ("device", "seed")):
        raise ValueError("Training and runtime device/seed must match")
    training = TrainingConfig(**{**training_input, "device": runtime["device"], "seed": runtime["seed"]})
    defaults = paper["post_training"]
    masking = _merge({"context_scale": defaults["context_scale"], "target_scale": defaults["target_scale"],
                       "num_target_blocks": defaults["target_blocks"], "target_aspect_ratio": defaults["target_aspect"],
                       "max_attempts": 32}, raw.get("masking", {}), "masking")
    # This constructs only a small CPU mask RNG; no model/image work occurs.
    BlockMaskSampler(model["image_size"] // model["patch_size"], seed=runtime["seed"], **masking)
    mae = _merge({"decoder_dim": model["predictor_dim"], "decoder_depth": model["predictor_depth"],
                   "num_heads": model["num_heads"], "norm_pix_loss": True, "mask_ratio": .75}, raw.get("mae", {}), "mae")
    for key in ("decoder_dim", "decoder_depth", "num_heads"):
        if type(mae[key]) is not int or mae[key] < 1:
            raise ValueError("MAE dimensions must be positive integers")
    if mae["decoder_dim"] % mae["num_heads"] or type(mae["norm_pix_loss"]) is not bool:
        raise ValueError("Invalid MAE decoder or normalized-pixel configuration")
    if type(mae["mask_ratio"]) not in (int, float) or not math.isfinite(mae["mask_ratio"]) or not 0 < mae["mask_ratio"] < 1:
        raise ValueError("MAE mask_ratio must be in (0,1)")
    if runtime["profile"] == "fixture":
        if (runtime["device"] != "cpu" or not 2 <= runtime["batch_size"] <= 4
                or model["image_size"] not in (32, 64) or model["in_channels"] != 3
                or model["embed_dim"] > 64 or model["predictor_dim"] > 64
                or model["depth"] > 2 or model["predictor_depth"] > 2
                or (model["image_size"] // model["patch_size"])**2 > 64
                or mae["decoder_dim"] > 64 or mae["decoder_depth"] > 2 or training.epochs > 2
                or masking["num_target_blocks"] > 6 or runtime["num_workers"] > 4 or runtime["num_threads"] > 4):
            raise ValueError("Fixtures require tiny CPU models, batch2-4 and at most two epochs")
    variant = raw.get("variant", f"baseline_{architecture}")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("variant must be a nonempty provenance label")
    return {"schema_version": 1, "architecture": architecture, "variant": variant,
            "runtime": runtime, "data": data, "model": model, "normalization": normalization,
            "training": asdict(training), "masking": masking, "mae": mae}


def load_pretrain_config(path):
    path = Path(path).resolve()
    return normalize_pretrain_config(json.loads(path.read_text()), base=path.parent)


def plan_pretraining(config):
    """Inspect local manifests/files only; do not decode images or allocate a ViT."""
    config = normalize_pretrain_config(config)
    data, runtime = config["data"], config["runtime"]
    info = inspect_manifest(data["unlabeled_manifest"], data["image_root"], labeled=False, require_files=True)
    if runtime["profile"] == "fixture" and info["num_samples"] > 16:
        raise ValueError("Fixture corpus is bounded to 16 images")
    return {"action": "plan", "architecture": config["architecture"], "profile": runtime["profile"],
            "device_requested": runtime["device"], "requires_explicit_execute": True,
            "model_constructed": False, "images_decoded": False, "training_run": False,
            "downloads_performed": False, "unlabeled_samples": info["num_samples"],
            "manifest_sha256": info["manifest_sha256"], "model": config["model"],
            "training": config["training"], "output_dir": runtime["output_dir"],
            "historical_experiment_match_verified": False}


def execute_pretraining(config, *, resume=False):
    from training.output import exclusive_training_output
    config=normalize_pretrain_config(config)
    with exclusive_training_output(config['runtime']['output_dir'],continuing=resume):
        return _execute_pretraining(config,resume=resume)


def _execute_pretraining(config, *, resume=False):
    """Run a prevalidated native job and save actual completed-epoch artifacts."""
    config = normalize_pretrain_config(config)
    plan = plan_pretraining(config)
    runtime, data, model_spec = config["runtime"], config["data"], config["model"]
    config_training = TrainingConfig(**config["training"])
    folder = Path(runtime["output_dir"])
    latest = folder / "latest.pt"
    if latest.exists() and not resume:
        raise FileExistsError("Baseline output exists; use --resume or a new output directory")
    from training.output import resume_checkpoint, restore_latest_pointer
    parent=resume_checkpoint(folder) if resume else None
    torch.set_num_threads(runtime["num_threads"])
    seed_runtime(runtime["seed"], runtime["device"])
    if config["architecture"] == "ijepa":
        model = build_full_ijepa(model_spec)
    else:
        keys = ("image_size", "patch_size", "in_channels", "embed_dim", "depth", "num_heads")
        encoder = MiniViT(**{key: model_spec[key] for key in keys})
        model = MAE(encoder, **{key: config["mae"][key] for key in ("decoder_dim", "decoder_depth", "num_heads", "norm_pix_loss")})
    dataset = UnlabeledManifestDataset(data["unlabeled_manifest"], data["image_root"],
                                       model_spec["image_size"], intensity=data["intensity"])
    batches = DataLoader(dataset, batch_size=runtime["batch_size"], shuffle=True, num_workers=runtime["num_workers"],
                         generator=torch.Generator(device="cpu").manual_seed(runtime["seed"]),
                         persistent_workers=False, drop_last=False)
    metadata_config = {**config, "runtime": {key: value for key, value in runtime.items() if key != "output_dir"},
                       "training": {key: value for key, value in config["training"].items()
                                    if key not in ("max_steps", "checkpoint_every")}}
    metadata = {"variant": config["variant"], "evidence": "synthetic_smoke" if runtime["profile"] == "fixture" else "measured_local",
                "mock": runtime["profile"] == "fixture", "source": "local_unlabeled_pretraining",
                "normalization": config["normalization"], "native_backbone": model_spec,
                "manifest_sha256": plan["manifest_sha256"], "unlabeled_samples": len(dataset),
                "job_configuration": metadata_config, "architecture_implementation": "project_native_" + config["architecture"],
                "historical_experiment_match_verified": False}
    sampler = BlockMaskSampler(model_spec["image_size"] // model_spec["patch_size"],
                                seed=runtime["seed"], **config["masking"])
    model, info = train_pretraining(model, batches, config_training, kind=config["architecture"],
                    mask_sampler=sampler, mask_ratio=config["mae"]["mask_ratio"],
                    normalizer=ImageNormalizer(config["normalization"], model_spec["in_channels"]),
                    checkpoint_dir=folder, resume_from=parent, metadata=metadata)
    if resume and not latest.exists(): restore_latest_pointer(folder,info['checkpoint'])
    result = {"action": "completed" if info["completed_epochs"] == config_training.epochs else "stopped_at_epoch_boundary",
              "architecture": config["architecture"],
              "evidence": metadata["evidence"], "paper_metrics_reproduced": False,
              "checkpoint": str(latest.resolve()), "checkpoint_sha256": checkpoint_sha256(latest), "training": info}
    temporary = folder / "pretraining_result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "pretraining_result.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly run the configured local training job")
    parser.add_argument("--resume", action="store_true", help="Continue latest.pt at a completed epoch boundary")
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute; default behavior is planning only")
    config = load_pretrain_config(args.config)
    result = execute_pretraining(config, resume=args.resume) if args.execute else plan_pretraining(config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
