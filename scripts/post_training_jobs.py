"""Offline full-checkpoint v4/v5/v6 jobs for the explicit experiment runner.

The architecture here is the project's native I-JEPA implementation. A supplied
full checkpoint must match it exactly; no official/raw weights are guessed or
downloaded. Raw images are corrupted before the declared mean/std normalizer.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import UnlabeledManifestDataset
from models import MiniViT, IJEPA, PostTrainingObjective, BlockMaskSampler, warm_start
from models.checkpoints import checkpoint_sha256
from training import TrainingConfig, load_training_model, train_post_training
from training.state import explicit_device, seed_runtime


_MODEL_FIELDS = {"mini", "image_size", "patch_size", "in_channels", "embed_dim", "depth",
                 "num_heads", "predictor_dim", "predictor_depth", "num_register_tokens"}


def build_full_ijepa(model_spec, registers=0):
    """Rebuild an exact native full I-JEPA for loading/training; never resolve weights.

    This helper allocates the requested architecture. Metadata-only planning must
    not call it. Fixtures are bounded by execute_post_training before allocation.
    """
    if not isinstance(model_spec, dict) or set(model_spec) - _MODEL_FIELDS:
        raise ValueError("Use the native full-I-JEPA model configuration")
    keys = ("image_size", "patch_size", "in_channels", "embed_dim", "depth", "num_heads")
    if any(type(model_spec.get(key)) is not int or model_spec[key] < 1
           for key in (*keys, "predictor_dim", "predictor_depth")):
        raise ValueError("Native model dimensions must be explicit positive integers")
    if type(registers) is not int or registers not in (0, 4):
        raise ValueError("Post-training uses zero or four register tokens")
    if model_spec["predictor_dim"] % model_spec["num_heads"]:
        raise ValueError("predictor_dim must divide into attention heads")
    encoder = MiniViT(**{key: model_spec[key] for key in keys}, num_register_tokens=registers)
    return IJEPA(encoder, predictor_dim=model_spec["predictor_dim"],
                 predictor_depth=model_spec["predictor_depth"], num_heads=model_spec["num_heads"])


class ImageNormalizer(nn.Module):
    """Serializable channel normalization applied after raw-image corruption."""
    def __init__(self, specification, in_channels=3):
        super().__init__()
        if not isinstance(specification, dict) or set(specification) != {"mean", "std"}:
            raise ValueError("Normalization requires explicit mean/std channel lists")
        mean = torch.as_tensor(specification["mean"], dtype=torch.float32)
        std = torch.as_tensor(specification["std"], dtype=torch.float32)
        if (mean.shape != (in_channels,) or std.shape != (in_channels,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Normalization needs finite means and positive channel standard deviations")
        self.register_buffer("mean", mean.reshape(1, -1, 1, 1))
        self.register_buffer("std", std.reshape(1, -1, 1, 1))

    def forward(self, images):
        return (images - self.mean.to(dtype=images.dtype)) / self.std.to(dtype=images.dtype)


def _checkpoint_metadata(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1 or not isinstance(payload.get("metadata"), dict):
        raise ValueError("A native full-I-JEPA checkpoint with provenance metadata is required")
    return copy.deepcopy(payload["metadata"])


def _has_fixture_ancestry(value):
    from models.provenance import has_synthetic_ancestry
    return has_synthetic_ancestry(value)


def _evidence_guard(metadata, profile, label):
    if profile == "experiment":
        if (metadata.get("mock") is not False or metadata.get("epoch_is_lineage_label") is not False
                or metadata.get("evidence") in ("synthetic_smoke", "synthetic", "fixture")
                or metadata.get("source") == "explicit_random_initialization"
                or _has_fixture_ancestry(metadata)):
            raise ValueError(f"Real {label} cannot use mock, synthetic or fixture-epoch checkpoints")
    elif metadata.get("evidence") != "synthetic_smoke" or metadata.get("mock") is not True:
        raise ValueError(f"Fixture {label} must explicitly declare synthetic_smoke and mock=true")


def _fixture_limits(runtime, data, post):
    if runtime["profile"] != "fixture":
        return
    model = post["backbone"]
    if (runtime["device"] != "cpu" or not 2 <= runtime["batch_size"] <= 4
            or data["image_size"] not in (32, 64) or model.get("image_size") not in (32, 64)
            or model.get("embed_dim", 0) > 64 or model.get("predictor_dim", 0) > 64
            or model.get("depth", 0) > 2 or model.get("predictor_depth", 0) > 2
            or runtime.get("num_workers", 0) > 4 or runtime.get("num_threads", 1) > 4
            or post["objective"]["target_blocks"] > 6 or post["objective"]["register_target_blocks"] > 6
            or model.get("in_channels") != 3 or model.get("patch_size", 0) < 1
            or (model["image_size"] // model["patch_size"])**2 > 64
            or any(post["epochs"][variant] > 2 for variant in post["variants"])):
        raise ValueError("Post-training fixtures require bounded mini CPU models and at most two epochs")


def execute_post_training(config, output_dir, *, resume=False, eval_only=False):
    from training.output import exclusive_training_output
    if not config['post_training']['enabled']: return []
    if resume and eval_only: raise ValueError('Choose resume or eval-only')
    with exclusive_training_output(output_dir,continuing=resume or eval_only):
        return _execute_post_training(config,output_dir,resume=resume,eval_only=eval_only)


def _execute_post_training(config, output_dir, *, resume=False, eval_only=False):
    """Execute selected variants in lineage order and return checkpoint records.

    ``output_dir/<variant>/epoch_0050.pt`` is generated only after 50 actual
    completed epochs. v5/v6 use that node or an explicit parent_checkpoints entry.
    Eval-only verifies and exposes existing complete checkpoints without fitting.
    Baseline warm starts load weights; within-variant resume restores all state.
    """
    post, runtime, data = config["post_training"], config["runtime"], config["data"]
    if not post["enabled"]:
        return []
    profile = runtime["profile"]
    if profile not in ("experiment", "fixture"):
        raise ValueError("Explicit experiment or fixture profile is required")
    device = explicit_device(runtime["device"])
    _fixture_limits(runtime, data, post)
    if post["backbone"]["image_size"] != data["image_size"]:
        raise ValueError("Unlabeled input resolution must match the full-I-JEPA model")
    variants = post["variants"]
    if not variants or len(set(variants)) != len(variants) or set(variants) - {"v4", "v5", "v6"}:
        raise ValueError("Declare distinct v4/v5/v6 variants")
    normalization = post.get("normalization", {"mean": [0.] * 3, "std": [1.] * 3})
    normalizer = ImageNormalizer(normalization, post["backbone"]["in_channels"])
    training = copy.deepcopy(post.get("training", {}))
    if set(training) - {item.name for item in fields(TrainingConfig)} or set(training) & {"epochs", "device", "seed"}:
        raise ValueError("post_training.training accepts TrainingConfig settings except epochs/device/seed")
    folder = Path(output_dir).resolve()
    job_spec = {"backbone": copy.deepcopy(post["backbone"]), "normalization": normalization,
                "objective": copy.deepcopy(post["objective"]),
                "training": {key: value for key, value in training.items() if key not in ("max_steps", "checkpoint_every")},
                "seed": runtime["seed"], "batch_size": runtime["batch_size"],
                "unlabeled_manifest": data["unlabeled_manifest"], "unlabeled_root": data["unlabeled_root"],
                "intensity": data["intensity"], "epochs": post["epochs"],
                "baseline_checkpoint": post["baseline_checkpoint"], "parent_checkpoints": post.get("parent_checkpoints", {}),
                "unlabeled_manifest_sha256": hashlib.sha256(Path(data["unlabeled_manifest"]).read_bytes()).hexdigest()}
    job_hash = hashlib.sha256(json.dumps(job_spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    dataset = None
    if not eval_only:
        dataset = UnlabeledManifestDataset(data["unlabeled_manifest"], data["unlabeled_root"],
                                           data["image_size"], intensity=data["intensity"])
        if profile == "fixture" and len(dataset) > 16:
            raise ValueError("Fixture post-training corpus is bounded to 16 images")
    records = []
    for variant in (name for name in ("v4", "v5", "v6") if name in variants):
        variant_dir = folder / variant
        latest = variant_dir / "latest.pt"
        epochs = post["epochs"][variant]
        train_config = TrainingConfig(**training, epochs=epochs, device=str(device), seed=runtime["seed"])
        if eval_only:
            checkpoint_metadata = _checkpoint_metadata(latest)
            _evidence_guard(checkpoint_metadata, profile, "evaluation")
            if (checkpoint_metadata.get("epoch") != epochs or checkpoint_metadata.get("job_spec_sha256") != job_hash
                    or checkpoint_metadata.get("training_spec", {}).get("config", {}).get("epochs") != epochs):
                raise ValueError("Eval-only requires a completed checkpoint for the configured post-training job")
            verified = build_full_ijepa(post["backbone"], registers=4 if variant == "v5" else 0)
            load_training_model(latest, verified, {"variant": variant})
            del verified
            records.append({"variant": variant, "checkpoint": str(latest), "checkpoint_sha256": checkpoint_sha256(latest), "metadata": checkpoint_metadata,
                            "training": {"completed_epochs": epochs, "actual_training_steps": checkpoint_metadata["actual_training_steps"],
                                         "evaluation_only": True}, "backbone": copy.deepcopy(post["backbone"]),
                            "normalization": normalization, "parent": checkpoint_metadata.get("parent")})
            continue
        if latest.exists() and not resume:
            raise FileExistsError(f"Post-training output already exists; use resume or a new output directory: {latest}")
        parent = post.get("parent_checkpoints", {}).get(variant)
        if parent is None:
            parent = post["baseline_checkpoint"] if variant == "v4" else folder / "v4" / "epoch_0050.pt"
        if parent is None or not Path(parent).is_file():
            raise FileNotFoundError(f"Missing local full checkpoint for {variant}: {parent}")
        parent_metadata = _checkpoint_metadata(parent)
        _evidence_guard(parent_metadata, profile, "warm start")
        if parent_metadata.get("normalization") is not None and parent_metadata["normalization"] != normalization:
            raise ValueError("Warm-start and child normalization differ")
        seed_runtime(runtime["seed"], device)
        model = build_full_ijepa(post["backbone"], registers=4 if variant == "v5" else 0)
        lineage = warm_start(parent, model, variant)
        objective_config = post["objective"]
        mode = {"v4": "noise", "v5": "register", "v6": "vicreg"}[variant]
        objective = PostTrainingObjective(mode, gaussian_sigma=objective_config["gaussian_sigma"],
                    poisson_peak=objective_config["poisson_rate"], jpeg_quality=objective_config["jpeg_quality"],
                    lambda_var=objective_config["lambda_var"], lambda_cov=objective_config["lambda_cov"],
                    normalizer=deepcopy_module(normalizer))
        register = variant == "v5"
        sampler = BlockMaskSampler(post["backbone"]["image_size"] // post["backbone"]["patch_size"], variant=mode,
                    seed=runtime["seed"], context_scale=objective_config["register_context_scale"] if register else objective_config["context_scale"],
                    target_scale=objective_config["target_scale"],
                    num_target_blocks=objective_config["register_target_blocks"] if register else objective_config["target_blocks"],
                    target_aspect_ratio=objective_config["register_target_aspect"] if register else objective_config["target_aspect"])
        batches = DataLoader(dataset, batch_size=runtime["batch_size"], shuffle=True,
                             generator=torch.Generator(device="cpu").manual_seed(runtime["seed"]),
                             num_workers=runtime["num_workers"], persistent_workers=False, drop_last=False)
        evidence = "synthetic_smoke" if profile == "fixture" else "measured_local"
        metadata = {"variant": variant, "evidence": evidence, "mock": profile == "fixture",
                    "source": "local_unlabeled_post_training", "parent": lineage,
                    "normalization": normalization, "native_backbone": post["backbone"],
                    "job_spec_sha256": job_hash, "unlabeled_manifest_sha256": dataset.manifest_info["manifest_sha256"],
                    "unlabeled_samples": len(dataset), "architecture_implementation": "project_native_ijepa",
                    "historical_experiment_match_verified": False}
        from training.output import resume_checkpoint, restore_latest_pointer
        resuming=None
        if resume and variant_dir.exists() and any(variant_dir.iterdir()):
            resuming=resume_checkpoint(variant_dir)
            _evidence_guard(_checkpoint_metadata(resuming), profile, "resume")
        model, info = train_post_training(model, objective, batches, train_config, mask_sampler=sampler,
                                          checkpoint_dir=variant_dir, resume_from=resuming,
                                          metadata=metadata)
        if resuming and not latest.exists(): restore_latest_pointer(variant_dir,info['checkpoint'])
        checkpoint_metadata = _checkpoint_metadata(latest)
        records.append({"variant": variant, "checkpoint": str(latest), "checkpoint_sha256": checkpoint_sha256(latest),
                        "metadata": checkpoint_metadata, "training": info,
                        "status": "complete" if info["completed_epochs"] == epochs else "stopped_at_epoch_boundary",
                        "backbone": copy.deepcopy(post["backbone"]),
                        "normalization": normalization, "parent": lineage})
        del model, objective, batches
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "post_training_manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"job_spec_sha256": job_hash, "records": records}, indent=2, allow_nan=False) + "\n")
    temporary.replace(manifest)
    return records


def deepcopy_module(module):
    """Each objective owns its normalizer buffers and explicit device placement."""
    return copy.deepcopy(module)
