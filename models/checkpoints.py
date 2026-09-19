"""Offline checkpoints with explicit provenance, strict loading and no downloads.

Native saves contain tensors and primitive metadata only and are read with
torch.load(weights_only=True). External model families require a caller-supplied
architecture factory and a local compatible checkpoint; random mocks are explicit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import copy
import hashlib
import torch
from torch import nn
from .vit_mini import MiniViT


class CheckpointError(ValueError):
    pass


def _primitive(value):
    if value is None or type(value) in (str, int, float, bool):
        return True
    if isinstance(value, (list, tuple)):
        return all(_primitive(v) for v in value)
    return isinstance(value, dict) and all(isinstance(k, str) and _primitive(v) for k, v in value.items())


def save_checkpoint(path, model: nn.Module, metadata: dict, optimizer=None) -> Path:
    if not isinstance(metadata, dict) or not _primitive(metadata):
        raise CheckpointError("Checkpoint metadata must contain primitive values only")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model_state_dict": model.state_dict(),
               "metadata": copy.deepcopy(metadata)}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local checkpoint does not exist: {path}. No automatic download is performed.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Could not safely load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise CheckpointError("Expected native checkpoint format_version=1; convert external weights explicitly")
    if not isinstance(payload.get("metadata"), dict) or not _primitive(payload["metadata"]):
        raise CheckpointError("Missing or invalid checkpoint metadata")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                              for k, v in state.items()):
        raise CheckpointError("Missing or invalid model_state_dict")
    return payload


def _check_metadata(metadata, expected_metadata):
    for key, expected in (expected_metadata or {}).items():
        if key not in metadata or metadata[key] != expected:
            raise CheckpointError(f"Checkpoint provenance mismatch for {key}: expected {expected!r}, "
                                  f"got {metadata.get(key)!r}")


def _check_state(model, state, allowed_missing=()):
    expected = model.state_dict()
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    forbidden = missing - set(allowed_missing)
    if forbidden or unexpected:
        raise CheckpointError(f"Checkpoint keys mismatch: missing={sorted(forbidden)}, unexpected={sorted(unexpected)}")
    mismatch = [key for key in state if state[key].shape != expected[key].shape]
    if mismatch:
        raise CheckpointError(f"Checkpoint tensor shape mismatch: {mismatch}")
    return missing


def load_checkpoint(path, model: nn.Module, expected_metadata: dict | None = None, optimizer=None) -> dict:
    """Restore a native checkpoint, rolling back both states if either load fails.

    Snapshots own their tensor storage: state_dict() alone aliases live parameters
    and optimizer buffers and would therefore be overwritten during a failed load.
    """
    payload = _read(path)
    _check_metadata(payload["metadata"], expected_metadata)
    _check_state(model, payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" not in payload:
        raise CheckpointError("Resume requested, but checkpoint has no optimizer state")
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    except Exception as exc:
        rollback_errors = []
        try:
            model.load_state_dict(model_before, strict=True)
        except Exception as rollback_exc:
            rollback_errors.append(f"model: {rollback_exc}")
        if optimizer is not None:
            try:
                optimizer.load_state_dict(optimizer_before)
            except Exception as rollback_exc:
                rollback_errors.append(f"optimizer: {rollback_exc}")
        if rollback_errors:
            raise CheckpointError("Checkpoint load failed and rollback was incomplete: "
                                  + "; ".join(rollback_errors)) from exc
        raise CheckpointError(f"Checkpoint load failed; previous states restored: {exc}") from exc
    return copy.deepcopy(payload["metadata"])


def warm_start(path, model: nn.Module, variant: str) -> dict:
    """Enforce v4 <- v3.1 ep201 and v5/v6 <- v4 ep50 (Sec. III-B).

Only v5 may add register parameters. All other keys and tensor dimensions must
match. New EMA register parameters are copied from the new context registers.
"""
    if variant not in ("v4", "v5", "v6"):
        raise CheckpointError("Warm-start variant must be v4, v5 or v6")
    payload = _read(path)
    expected = {"variant": "v3.1" if variant == "v4" else "v4",
                "epoch": 201 if variant == "v4" else 50}
    _check_metadata(payload["metadata"], expected)
    state = payload["model_state_dict"]
    allowed = [key for key in model.state_dict() if key.endswith("register_tokens")] if variant == "v5" else []
    missing = _check_state(model, state, allowed_missing=allowed)
    if variant == "v5":
        encoder = getattr(model, "context_encoder", model)
        if getattr(encoder, "num_register_tokens", 0) != 4:
            raise CheckpointError("v5 requires exactly four register tokens")
    merged = model.state_dict()
    merged.update(state)
    for key in missing:
        if key.startswith("target_encoder."):
            context_key = "context_encoder." + key[len("target_encoder."):]
            if context_key in merged:
                merged[key] = merged[context_key].clone()
    model.load_state_dict(merged, strict=True)
    return {"variant": variant, "warm_start": copy.deepcopy(payload["metadata"]),
            "source_path": str(Path(path).resolve()), "source_sha256": checkpoint_sha256(path),
            "new_parameters": sorted(missing)}


def checkpoint_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_encoder(name="mini", *, checkpoint=None, use_random_init=False, factory: Callable | None = None,
                   expected_metadata=None, **kwargs):
    """Build offline encoder; EVA-X/RAD-DINO use explicit factories or labelled mocks.

This does not claim that a MiniViT can load an EVA-X or RAD-DINO checkpoint. A
factory must instantiate the exact architecture and native conversion must record
the source. Set use_random_init=True only for tests/scaffolding, never paper scores.
    """
    name = name.lower().replace("-", "_")
    if name not in ("mini", "ijepa", "mae", "eva_x", "rad_dino"):
        raise ValueError(f"Unknown encoder: {name}")
    if checkpoint is None and not use_random_init:
        raise CheckpointError("A local checkpoint is required; use_random_init=True explicitly enables a mock")
    if name in ("eva_x", "rad_dino") and factory is None and checkpoint is not None:
        raise CheckpointError(f"{name} requires a factory for the exact external architecture")
    model = factory(**kwargs) if factory is not None else MiniViT(**kwargs)
    if checkpoint is not None:
        metadata = load_checkpoint(checkpoint, model, expected_metadata=expected_metadata)
        model.provenance = {"source": "local_checkpoint", "requested_family": name,
                            "checkpoint_sha256": checkpoint_sha256(checkpoint), "metadata": metadata,
                            "mock": bool(metadata.get("mock", False))}
    else:
        model.provenance = {"source": "explicit_random_initialization", "requested_family": name,
                            "mock": True, "is_paper_pretrained_model": False}
    return model
