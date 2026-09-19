"""Local, native-compatible epoch checkpoints with explicit RNG restoration.

No device selection, downloads, or model construction occurs here. Checkpoints
are safe-loadable tensor/primitive dictionaries. Resumption is supported at a
completed epoch boundary on the same device and with the same named generators.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


class TrainingStateError(ValueError):
    """The requested checkpoint cannot safely resume this training run."""


def explicit_device(device="cpu"):
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("Training supports explicitly selected cpu or cuda:N devices")
    if result.type == "cuda" and result.index is None:
        raise ValueError("Specify a CUDA device index explicitly, for example cuda:0")
    return result


def seed_runtime(seed, device="cpu"):
    """Seed CPU/Python/NumPy and only an explicitly selected CUDA device."""
    device = explicit_device(device)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # CPU generator only: torch.manual_seed also initializes accelerator RNGs.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _primitive(value):
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_primitive(item) for item in value)
    return isinstance(value, dict) and all(isinstance(k, str) and _primitive(v) for k, v in value.items())


def collect_generators(batches, generators=None, **owned):
    """Capture loader/sampler generators without advancing them.

    Persistent worker processes retain inaccessible dataset RNG state; use
    deterministic data or newly seeded workers per epoch when checkpointing.
    """
    result = dict(generators or {})
    if getattr(batches, "persistent_workers", False):
        raise ValueError("Epoch-resumable training requires persistent_workers=False")
    automatic = {"dataloader": getattr(batches, "generator", None),
                 "sampler": getattr(getattr(batches, "sampler", None), "generator", None), **owned}
    for name, generator in automatic.items():
        if generator is None:
            continue
        if name in result and result[name] is not generator:
            raise ValueError(f"Conflicting named training generator: {name}")
        result[name] = generator
    if any(not isinstance(k, str) or not isinstance(v, torch.Generator) for k, v in result.items()):
        raise ValueError("generators must map names to torch.Generator instances")
    return result


def _rng_state(device):
    numpy_state = np.random.get_state()
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
             "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                       "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                       "cached_gaussian": float(numpy_state[4])}, "device": str(device)}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    value = state["numpy"]
    np.random.set_state((value["algorithm"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def save_training_state(path, model, optimizer, scheduler, epoch, history, metadata=None,
                        generators=None, global_step=0, *, device="cpu"):
    """Atomically save a completed epoch; native model loaders remain compatible.

    Evidence/mock/source fields are supplied by the caller and never upgraded.
    ``epoch`` is actual completed work, not a paper-lineage fixture label.
    """
    device = explicit_device(device)
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise ValueError("epoch and global_step must be nonnegative integers")
    metadata = copy.deepcopy(metadata or {})
    if not _primitive(metadata) or not _primitive(history):
        raise ValueError("Checkpoint metadata/history must contain finite primitive values")
    metadata.update(epoch=epoch, epoch_is_lineage_label=False,
                    actual_training_steps=global_step, global_step=global_step)
    generators = dict(generators or {})
    payload = {"format_version": 1, "training_state_version": 1,
               "model_state_dict": model.state_dict(), "metadata": metadata,
               "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
               "epoch": epoch, "next_epoch": epoch, "global_step": global_step,
               "history": copy.deepcopy(history), "rng_state": _rng_state(device),
               "generator_states": {key: {"device": str(gen.device), "state": gen.get_state()}
                                    for key, gen in generators.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _read(path):
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TrainingStateError(f"Could not read local training checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise TrainingStateError("Expected native format_version=1 checkpoint")
    if not isinstance(payload.get("metadata"), dict) or not _primitive(payload["metadata"]):
        raise TrainingStateError("Invalid checkpoint metadata")
    return payload


def _validate_model(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise TrainingStateError("Checkpoint model keys do not match the requested architecture")
    if any(not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape
           for key, tensor in state.items()):
        raise TrainingStateError("Checkpoint model tensor shapes do not match")


def _check_metadata(metadata, expected):
    for key, value in (expected or {}).items():
        if metadata.get(key) != value or key not in metadata:
            raise TrainingStateError(f"Checkpoint metadata mismatch for {key}")


def load_training_model(path, model, expected_metadata=None):
    """Load complete model/head/BN tensors for evaluation without changing RNG."""
    payload = _read(path)
    _check_metadata(payload["metadata"], expected_metadata)
    _validate_model(model, payload.get("model_state_dict"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return copy.deepcopy(payload["metadata"])


def load_training_state(path, model, optimizer, scheduler, *, generators=None,
                        expected_metadata=None, device="cpu"):
    """Restore model, optimization state and RNG, returning a zero-based next_epoch.

    Validate before mutation and roll back failed restoration. Rollback snapshots
    live on CPU to avoid doubling accelerator model storage. Exact continuation
    requires unchanged configuration, sample order and deterministic data loading.
    """
    device = explicit_device(device)
    payload = _read(path)
    if payload.get("training_state_version") != 1:
        raise TrainingStateError("Checkpoint lacks a resumable training_state_version=1")
    _check_metadata(payload["metadata"], expected_metadata)
    _validate_model(model, payload.get("model_state_dict"))
    epoch, step = payload.get("epoch"), payload.get("global_step")
    if type(epoch) is not int or epoch < 0 or payload.get("next_epoch") != epoch:
        raise TrainingStateError("Checkpoint must describe a completed epoch boundary")
    if type(step) is not int or step < 0 or not isinstance(payload.get("history"), list):
        raise TrainingStateError("Invalid checkpoint history or optimizer-step count")
    if len(payload["history"]) != epoch or not _primitive(payload["history"]):
        raise TrainingStateError("History must contain one finite record per completed epoch")
    if payload["metadata"].get("epoch") != epoch or payload["metadata"].get("actual_training_steps") != step:
        raise TrainingStateError("Checkpoint metadata does not match actual completed work")
    if (scheduler is None) != (payload.get("scheduler_state_dict") is None):
        raise TrainingStateError("Checkpoint scheduler does not match the requested schedule")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise TrainingStateError("Checkpoint has no optimizer state")
    generators = dict(generators or {})
    saved_generators = payload.get("generator_states", {})
    if set(generators) != set(saved_generators):
        raise TrainingStateError("Named RNG generators differ from the checkpoint")
    if any(str(generator.device) != saved_generators[name]["device"]
           for name, generator in generators.items()):
        raise TrainingStateError("Generator devices differ from the checkpoint")
    if payload.get("rng_state", {}).get("device") != str(device):
        raise TrainingStateError("Exact resume requires the checkpoint's explicit device")
    before = {"model": _cpu_snapshot(model.state_dict()),
              "optimizer": _cpu_snapshot(optimizer.state_dict()),
              "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler else None,
              "rng": _rng_state(device), "generators": {k: v.get_state() for k, v in generators.items()}}
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        _restore_rng(payload["rng_state"], device)
        for name, generator in generators.items():
            generator.set_state(saved_generators[name]["state"].cpu())
    except Exception as exc:
        model.load_state_dict(before["model"], strict=True)
        optimizer.load_state_dict(before["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(before["scheduler"])
        _restore_rng(before["rng"], device)
        for name, generator in generators.items():
            generator.set_state(before["generators"][name])
        raise TrainingStateError(f"Checkpoint restore failed; previous states restored: {exc}") from exc
    return {"next_epoch": epoch, "global_step": step, "history": copy.deepcopy(payload["history"]),
            "metadata": copy.deepcopy(payload["metadata"])}
