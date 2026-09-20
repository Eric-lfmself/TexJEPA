"""Declarative, strictly local image backbones. No implicit downloads or factories.

The native backend is this project's own MiniViT at configurable dimensions, not
an implementation of any pretrained model family. Official code is used only from
an explicitly trusted local repository. See docs/BACKBONES.md for source evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from argparse import Namespace
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoints import CheckpointError, checkpoint_sha256
from .vit_mini import MiniViT, gather_tokens, validate_indices


_RECIPES = {
    "ijepa_huge_patch14": {
        "backend": "local_factory", "family": "ijepa", "module": "src.models.vision_transformer",
        "factory": "vit_huge", "image_size": 224, "patch_size": 14,
        "kwargs": {"img_size": [224], "patch_size": 14},
        "interface": {"kind": "ijepa", "prefix_tokens": 0, "cls_index": None}},
    "mae_huge_patch14": {
        "backend": "local_factory", "family": "mae", "module": "models_mae",
        "factory": "mae_vit_huge_patch14", "image_size": 224, "patch_size": 14,
        "kwargs": {"img_size": 224},
        "interface": {"kind": "mae", "prefix_tokens": 1, "cls_index": 0,
                      "embed_dim_path": "pos_embed"}},
    "eva_x_base_patch16": {
        "backend": "local_factory", "family": "eva_x", "module": "eva_x", "factory": "EVA_X",
        "image_size": 224, "patch_size": 16, "pooling": "model_head",
        "kwargs": {"img_size": 224, "patch_size": 16, "embed_dim": 768, "depth": 12,
                   "num_heads": 12, "qkv_fused": False, "mlp_ratio": 8 / 3,
                   "swiglu_mlp": True, "scale_mlp": True, "use_rot_pos_emb": True,
                   "ref_feat_shape": [14, 14], "num_classes": 0, "global_pool": "avg"},
        "interface": {"kind": "tokens", "prefix_tokens": 1, "cls_index": 0,
                      "norm_paths": ["norm", "fc_norm"]}},
    "rad_dino": {"backend": "huggingface", "family": "rad_dino", "image_size": 518,
                 "patch_size": 14, "pooling": "cls"},
}
_FIELDS = {"backend", "family", "recipe", "architecture", "kwargs", "image_size", "patch_size",
           "in_channels", "normalization", "pooling", "checkpoint", "use_random_init", "repo_path",
           "module", "factory", "trust_local_code", "interface", "model_path", "provenance_note"}
_INTERFACE_FIELDS = {"kind", "method", "output_key", "prefix_tokens", "cls_index", "blocks_path",
                     "norm_paths", "embed_dim_path", "mask_argument", "mask_as_list"}
_FORBIDDEN_KWARGS = {"pretrained", "checkpoint_path", "pretrained_cfg", "pretrained_cfg_overlay",
                     "pretrained_backbone", "weights", "weights_path", "trust_remote_code",
                     "local_files_only", "force_download", "device_map"}


def _primitive(value):
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_primitive(x) for x in value)
    return isinstance(value, dict) and all(isinstance(k, str) and _primitive(v) for k, v in value.items())


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _local_path(value, name, *, check_files=False, directory=False):
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError(f"{name} must be an explicit local filesystem path")
    path = Path(value).expanduser()
    if check_files and not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"Missing local {name}: {path}; no automatic download is performed")
    return str(path.resolve())


def validate_backbone_spec(spec: dict, check_files: bool = False) -> dict:
    """Return a copied normalized config, without imports or model allocation.

File checks are opt-in so dry-run plans can describe user-provided future paths.
All external initialization requires existing weights unless random init is explicit.
"""
    if not isinstance(spec, dict) or not _primitive(spec):
        raise ValueError("backbone spec must be a dictionary of finite JSON values")
    if set(spec) - _FIELDS:
        raise ValueError(f"Unknown backbone fields: {sorted(set(spec) - _FIELDS)}")
    spec = copy.deepcopy(spec)
    if "recipe" in spec:
        if spec["recipe"] not in _RECIPES:
            raise ValueError(f"Unknown backbone recipe: {spec['recipe']}")
        defaults = copy.deepcopy(_RECIPES[spec["recipe"]])
        for key in ("kwargs", "interface"):
            if key in spec and not isinstance(spec[key], dict):
                raise ValueError(f"{key} must be an object")
            if key in defaults:
                spec[key] = {**defaults[key], **spec.get(key, {})}
        spec = {**defaults, **spec}
    backend = spec.setdefault("backend", "native")
    if backend not in ("native", "local_factory", "timm", "huggingface"):
        raise ValueError("backend must be native, local_factory, timm or huggingface")
    family = spec.setdefault("family", "native" if backend == "native" else "custom")
    if family not in ("native", "custom", "ijepa", "mae", "eva_x", "rad_dino"):
        raise ValueError("Unsupported backbone family")
    if backend == "native" and family != "native":
        raise ValueError("Native MiniViT cannot be labelled as an official pretrained family")
    if backend == "native" and spec.get("architecture", "MiniViT") != "MiniViT":
        raise ValueError("The native architecture is MiniViT at configurable dimensions")
    if backend == "timm" and family != "custom":
        raise ValueError("Generic timm models require family=custom; use an exact official local recipe")
    if backend == "huggingface" and family not in ("custom", "rad_dino"):
        raise ValueError("Hugging Face backend supports local DINOv2/RAD-DINO snapshots only")
    kwargs = spec.setdefault("kwargs", {})
    if not isinstance(kwargs, dict) or set(kwargs) & _FORBIDDEN_KWARGS:
        raise ValueError("kwargs must be an object without pretrained/download/checkpoint overrides")
    for name, default in (("image_size", kwargs.get("image_size", 32 if backend == "native" else 224)),
                          ("patch_size", kwargs.get("patch_size", 8 if backend == "native" else 14)),
                          ("in_channels", kwargs.get("in_channels", 3))):
        _positive_int(spec.setdefault(name, default), name)
    if spec["image_size"] % spec["patch_size"]:
        raise ValueError("image_size must be divisible by patch_size")
    for argument, field in (("img_size", "image_size"), ("in_chans", "in_channels")):
        if argument in kwargs:
            value = kwargs[argument]
            values = value if isinstance(value, list) else [value]
            if not values or any(type(v) is not int or v != spec[field] for v in values):
                raise ValueError(f"kwargs.{argument} must agree with {field}")
    if "patch_size" in kwargs and kwargs["patch_size"] != spec["patch_size"]:
        raise ValueError("kwargs.patch_size must agree with patch_size")
    if backend == "native":
        allowed = {"image_size", "patch_size", "in_channels", "embed_dim", "depth", "num_heads", "num_register_tokens"}
        if set(kwargs) - allowed:
            raise ValueError(f"Unknown native architecture kwargs: {sorted(set(kwargs) - allowed)}")
        for key in ("image_size", "patch_size", "in_channels"):
            if key in kwargs and kwargs[key] != spec[key]:
                raise ValueError(f"Conflicting native {key}")
            kwargs[key] = spec[key]
        for key, default in (("embed_dim", 64), ("depth", 2), ("num_heads", 4)):
            _positive_int(kwargs.setdefault(key, default), key)
        if kwargs["embed_dim"] % kwargs["num_heads"]:
            raise ValueError("embed_dim must be divisible by num_heads")
        if type(kwargs.setdefault("num_register_tokens", 0)) is not int or kwargs["num_register_tokens"] < 0:
            raise ValueError("num_register_tokens must be a nonnegative integer")
    normalization = spec.setdefault("normalization", {"mean": [0.] * spec["in_channels"],
                                                       "std": [1.] * spec["in_channels"]} if backend == "native" else None)
    if not isinstance(normalization, dict) or set(normalization) != {"mean", "std"}:
        raise ValueError("Explicit normalization={mean:[...], std:[...]} is required for external backbones")
    for name in ("mean", "std"):
        values = normalization[name]
        if not isinstance(values, list) or len(values) != spec["in_channels"] or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("Normalization must contain one finite number per input channel")
    if any(x <= 0 for x in normalization["std"]):
        raise ValueError("Normalization standard deviations must be positive")
    pooling = spec.setdefault("pooling", "cls" if backend == "huggingface" else "mean_patch")
    if pooling not in ("mean_patch", "cls", "model_head"):
        raise ValueError("pooling must be mean_patch, cls or model_head")
    if type(spec.setdefault("use_random_init", False)) is not bool:
        raise ValueError("use_random_init must be a boolean")
    checkpoint = spec.get("checkpoint")
    if isinstance(checkpoint, str):
        checkpoint = spec["checkpoint"] = {"path": checkpoint}
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or set(checkpoint) - {"path", "state_key", "strip_prefix", "conversion", "sha256"}:
            raise ValueError("Invalid checkpoint fields")
        checkpoint["path"] = _local_path(checkpoint.get("path"), "checkpoint", check_files=check_files)
        for key in ("state_key", "strip_prefix"):
            if not isinstance(checkpoint.setdefault(key, ""), str):
                raise ValueError(f"checkpoint.{key} must be a string")
        if checkpoint.setdefault("conversion", "none") not in ("none", "eva_x_official"):
            raise ValueError("Unknown checkpoint conversion")
        if checkpoint["conversion"] == "eva_x_official" and family != "eva_x":
            raise ValueError("EVA-X conversion is restricted to an exact EVA-X local architecture")
        if "sha256" in checkpoint and (not isinstance(checkpoint["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])):
            raise ValueError("checkpoint.sha256 must be 64 lowercase hexadecimal digits")
    if backend == "huggingface":
        if checkpoint is not None or spec["use_random_init"] or kwargs:
            raise ValueError("Hugging Face requires one existing model_path snapshot; no checkpoint/kwargs/random override")
        spec["model_path"] = _local_path(spec.get("model_path"), "model_path", check_files=check_files, directory=True)
    elif checkpoint is None and not spec["use_random_init"]:
        raise ValueError("A local checkpoint is required, or explicitly set use_random_init=true")
    if backend == "local_factory":
        if spec.get("trust_local_code") is not True:
            raise ValueError("Local source execution requires trust_local_code=true for your explicit repo_path")
        spec["repo_path"] = _local_path(spec.get("repo_path"), "repo_path", check_files=check_files, directory=True)
        for name in ("module", "factory"):
            if not isinstance(spec.get(name), str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", spec[name]):
                raise ValueError(f"{name} must be a local Python dotted name")
        if check_files:
            module = Path(spec["repo_path"]).joinpath(*spec["module"].split("."))
            if not module.with_suffix(".py").is_file() and not (module / "__init__.py").is_file():
                raise FileNotFoundError(f"Local factory module does not exist: {module}")
    if backend == "timm":
        if not isinstance(spec.get("architecture"), str) or not re.fullmatch(r"[A-Za-z0-9_.]+", spec["architecture"]):
            raise ValueError("timm architecture must be a local registry name, never a hub URL or identifier")
    interface = spec.setdefault("interface", {})
    if not isinstance(interface, dict) or set(interface) - _INTERFACE_FIELDS:
        raise ValueError("Invalid backbone interface fields")
    kind = interface.setdefault("kind", "native" if backend == "native" else "huggingface" if backend == "huggingface" else "tokens")
    if kind not in ("native", "ijepa", "mae", "tokens", "huggingface"):
        raise ValueError("Unknown token interface kind")
    if (backend == "native") != (kind == "native") or (backend == "huggingface") != (kind == "huggingface"):
        raise ValueError("Native/Hugging Face interface must match its backend")
    for key, value in {"method": "forward_features", "output_key": None, "prefix_tokens": 0 if kind in ("native", "ijepa") else 1,
                       "cls_index": None if kind in ("native", "ijepa") else 0,
                       "blocks_path": "encoder.layer" if kind == "huggingface" else "blocks",
                       "norm_paths": ["layernorm"] if kind == "huggingface" else ["norm"],
                       "embed_dim_path": "config.hidden_size" if kind == "huggingface" else "embed_dim",
                       "mask_argument": None, "mask_as_list": False}.items():
        interface.setdefault(key, value)
    if type(interface["prefix_tokens"]) is not int or interface["prefix_tokens"] < 0:
        raise ValueError("prefix_tokens must be a nonnegative integer")
    cls_index = interface["cls_index"]
    if cls_index is not None and (type(cls_index) is not int or not 0 <= cls_index < interface["prefix_tokens"]):
        raise ValueError("cls_index must identify a prefix token or be null")
    if pooling == "cls" and cls_index is None:
        raise ValueError("CLS pooling requires a declared CLS token")
    for key in ("method", "blocks_path", "embed_dim_path"):
        if not isinstance(interface[key], str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", interface[key]):
            raise ValueError(f"interface.{key} must be a dotted attribute name")
    if not isinstance(interface["norm_paths"], list) or not interface["norm_paths"] or any(
            not isinstance(x, str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", x) for x in interface["norm_paths"]):
        raise ValueError("norm_paths must list final normalization modules")
    if interface["output_key"] is not None and not isinstance(interface["output_key"], str):
        raise ValueError("output_key must be a string or null")
    if interface["mask_argument"] is not None and (not isinstance(interface["mask_argument"], str) or not interface["mask_argument"].isidentifier()):
        raise ValueError("mask_argument must be a keyword name or null")
    if type(interface["mask_as_list"]) is not bool:
        raise ValueError("mask_as_list must be a boolean")
    if kind != "tokens" and interface["mask_argument"] is not None:
        raise ValueError("mask_argument overrides apply only to the generic tokens interface")
    if kind in ("native", "ijepa") and (interface["prefix_tokens"] or cls_index is not None):
        raise ValueError("Native and I-JEPA token outputs contain patches only")
    if kind == "mae" and (interface["prefix_tokens"] != 1 or cls_index != 0):
        raise ValueError("Official MAE encoder requires one leading CLS token")
    if pooling == "model_head" and kind in ("native", "ijepa", "mae", "huggingface"):
        raise ValueError("model_head pooling requires a token model exposing forward_head(pre_logits=True)")
    return spec


def _attribute(obj, path):
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


@contextmanager
def _local_import_path(repo):
    """Keep official absolute imports local and reject existing namespace collisions."""
    root = Path(repo).resolve()
    top_names = {p.stem if p.is_file() else p.name for p in root.iterdir()
                 if p.suffix == ".py" or (p.is_dir() and p.name.isidentifier())}
    for name in top_names:
        loaded = sys.modules.get(name)
        origin = getattr(loaded, "__file__", None) if loaded is not None else None
        locations = [origin] if origin else list(getattr(loaded, "__path__", ()))
        if loaded is not None and (not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations)):
            raise ValueError(f"Local source namespace collision: {name}; use a fresh process for this repository")
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))


def _create_local(spec):
    with _local_import_path(spec["repo_path"]):
        importlib.invalidate_caches()
        module = importlib.import_module(spec["module"])
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(Path(spec["repo_path"])):
            raise ValueError("Factory resolved outside the explicitly trusted local source repository")
        model = _attribute(module, spec["factory"])(**spec["kwargs"])
    source_files = []
    for directory, dirs, files in os.walk(spec["repo_path"], followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", ".venv", "venv", "__pycache__")
                         and not Path(directory, d).is_symlink())
        source_files.extend(Path(directory, f) for f in sorted(files) if f.endswith(".py"))
    source_manifest = {str(p.relative_to(spec["repo_path"])): checkpoint_sha256(p)
                       for p in source_files if not p.is_symlink()}
    source_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    return model, {"source_file": str(origin), "source_file_sha256": checkpoint_sha256(origin),
                   "source_python_tree_sha256": source_hash, "source_python_file_count": len(source_manifest)}


# Adapted from EVA-X checkpoint_filter_fn (Apache-2.0); see THIRD_PARTY_NOTICES.md.
# Modified to convert keys only, record changes, and reject conversion collisions.
def _eva_x_convert(state):
    """Explicit key-only subset of hustvl/EVA-X/eva_x.py checkpoint_filter_fn.

No interpolation/resampling occurs: changed patch grids or dimensions fail strict
loading. The published unused MIM head/mask and deterministic RoPE are recorded.
"""
    converted, changes, dropped = {}, {}, []
    mim = "mask_token" in state
    split_qkv = "blocks.0.attn.q_proj.weight" in state
    replacements = (("mlp.ffn_ln", "mlp.norm"), ("attn.inner_attn_ln", "attn.norm"),
                    ("mlp.w12", "mlp.fc1"), ("mlp.w1", "mlp.fc1_g"),
                    ("mlp.w2", "mlp.fc1_x"), ("mlp.w3", "mlp.fc2"))
    for key, tensor in state.items():
        if key.startswith("rope.") or ".rope." in key or (mim and key in ("mask_token", "lm_head.weight", "lm_head.bias")):
            dropped.append(key)
            continue
        target = key
        for old, new in replacements:
            target = target.replace(old, new)
        if split_qkv:
            target = target.replace("q_bias", "q_proj.bias").replace("v_bias", "v_proj.bias")
        if mim and target in ("norm.weight", "norm.bias"):
            target = target.replace("norm", "fc_norm", 1)
        if target in converted:
            raise CheckpointError(f"Checkpoint conversion key collision: {target}")
        converted[target] = tensor
        if target != key:
            changes[key] = target
    return converted, {"renamed_keys": changes, "discarded_pretraining_or_rope_keys": sorted(dropped),
                       "interpolation": False}


def _load_external_checkpoint(model, checkpoint):
    path = checkpoint["path"]
    digest = checkpoint_sha256(path)
    if checkpoint.get("sha256", digest) != digest:
        raise CheckpointError("Checkpoint SHA256 does not match the configured source artifact")
    if Path(path).suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Local safetensors loading requires the already-installed safetensors package") from exc
        state = load_file(path, device="cpu")
    else:
        try:
            # Official MAE training files include args=argparse.Namespace. This
            # inert standard-library container is the only added safe global;
            # arbitrary source classes and unrestricted pickle remain forbidden.
            with torch.serialization.safe_globals([Namespace]):
                state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointError(f"Cannot safely read local backbone checkpoint: {exc}") from exc
    for key in checkpoint["state_key"].split(".") if checkpoint["state_key"] else []:
        if not isinstance(state, dict) or key not in state:
            raise CheckpointError(f"Configured state_key does not exist: {checkpoint['state_key']}")
        state = state[key]
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()):
        raise CheckpointError("Checkpoint selection must contain only string-to-tensor state entries")
    prefix = checkpoint["strip_prefix"]
    if prefix:
        if any(not key.startswith(prefix) for key in state):
            raise CheckpointError("strip_prefix must match every selected checkpoint key; no silent filtering")
        state = {key[len(prefix):]: value for key, value in state.items()}
    conversion = {}
    if checkpoint["conversion"] == "eva_x_official":
        state, conversion = _eva_x_convert(state)
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    shapes = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if missing or unexpected or shapes:
        raise CheckpointError(f"Strict backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={sorted(shapes)}")
    model.load_state_dict(state, strict=True)
    return {"checkpoint_path": path, "checkpoint_sha256": digest, "state_key": checkpoint["state_key"],
            "strip_prefix": prefix, "conversion": checkpoint["conversion"], **conversion}


def _create_huggingface(spec):
    root = Path(spec["model_path"])
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Local Hugging Face snapshot requires config.json")
    config = json.loads(config_path.read_text())
    if config.get("model_type") not in ("dinov2", "dinov2_with_registers"):
        raise ValueError("Local Hugging Face backbone must use built-in DINOv2, without remote model code")
    if config.get("auto_map"):
        raise ValueError("Snapshots requesting custom/remote model code are not accepted")
    expected_registers = config.get("num_register_tokens", 0)
    if type(expected_registers) is not int or expected_registers < 0:
        raise ValueError("DINOv2 num_register_tokens must be a nonnegative integer")
    if config["model_type"] == "dinov2" and expected_registers:
        raise ValueError("Register tokens require the built-in dinov2_with_registers model type")
    expected_prefix = 1 + expected_registers
    interface = spec["interface"]
    # The normalized spec is generated before reading a snapshot; set the actual
    # built-in DINOv2 register count and preserve it in the final provenance.
    if interface["prefix_tokens"] not in (1, expected_prefix) or interface["cls_index"] != 0:
        raise ValueError("Hugging Face prefix/CLS layout conflicts with the local DINOv2 config")
    interface["prefix_tokens"], interface["cls_index"] = expected_prefix, 0
    if "patch_size" in config and config["patch_size"] != spec["patch_size"]:
        raise ValueError("Configured patch_size differs from the local Hugging Face architecture")
    if "num_channels" in config and config["num_channels"] != spec["in_channels"]:
        raise ValueError("Configured in_channels differs from the local Hugging Face architecture")
    weight_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".safetensors", ".bin"))
    if not weight_files:
        raise FileNotFoundError("Local Hugging Face snapshot has no model weights")
    try:
        import transformers
    except ImportError as exc:
        raise ImportError("Hugging Face backend requires an already-installed compatible transformers package") from exc
    model, info = transformers.AutoModel.from_pretrained(str(root), local_files_only=True, trust_remote_code=False,
                                                        token=False, force_download=False, weights_only=True,
                                                        ignore_mismatched_sizes=False, output_loading_info=True)
    defects = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if defects:
        raise CheckpointError(f"Hugging Face checkpoint did not load strictly: {defects}")
    files = sorted(set(weight_files + list(root.glob("*.json"))))
    manifest = {p.name: checkpoint_sha256(p) for p in files}
    return model, {"model_path": str(root), "local_artifact_sha256": manifest,
                   "local_files_only": True, "trust_remote_code": False, "model_type": config["model_type"],
                   "transformers_version": getattr(transformers, "__version__", "unknown")}


class BackboneAdapter(nn.Module):
    """BCHW float input; normalization only, with no resize, rescale or clipping."""
    def __init__(self, model: nn.Module, spec: dict, provenance: dict):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("Backbone constructor must return torch.nn.Module")
        self.model, self.spec = model, copy.deepcopy(spec)
        self.image_size, self.patch_size = spec["image_size"], spec["patch_size"]
        self.in_channels = spec["in_channels"]
        self.grid_size, self.num_patches = self.image_size // self.patch_size, (self.image_size // self.patch_size) ** 2
        self.interface = copy.deepcopy(spec["interface"])
        dim = _attribute(model, self.interface["embed_dim_path"])
        self.embed_dim = dim.shape[-1] if isinstance(dim, Tensor) else dim
        _positive_int(self.embed_dim, "model embed_dim")
        if not isinstance(self.blocks, (nn.ModuleList, nn.Sequential)) or not len(self.blocks):
            raise ValueError("blocks_path must expose a nonempty ordered ModuleList or Sequential")
        if any(not isinstance(_attribute(model, path), nn.Module) for path in self.interface["norm_paths"]):
            raise ValueError("norm_paths must resolve to normalization modules")
        self.num_register_tokens = getattr(model, "num_register_tokens", max(0, self.interface["prefix_tokens"] - (self.interface["cls_index"] is not None)))
        self.register_buffer("input_mean", torch.tensor(spec["normalization"]["mean"], dtype=torch.float32).reshape(1, -1, 1, 1))
        self.register_buffer("input_std", torch.tensor(spec["normalization"]["std"], dtype=torch.float32).reshape(1, -1, 1, 1))
        self.supports_masked_tokens = self.interface["kind"] in ("native", "ijepa", "mae") or self.interface["mask_argument"] is not None
        self.provenance = {**copy.deepcopy(provenance), "backend": spec["backend"], "requested_family": spec["family"],
                           "architecture_class": f"{type(model).__module__}.{type(model).__qualname__}",
                           "backbone_spec": copy.deepcopy(spec), "pooling": spec["pooling"],
                           "normalization": copy.deepcopy(spec["normalization"]), "input_clipping": False,
                           "patch_tokens_exclude_prefix_tokens": self.interface["prefix_tokens"],
                           "family_is_user_declared": spec["backend"] != "native",
                           "is_paper_pretrained_model": False, "historical_experiment_match_verified": False}

    @property
    def blocks(self):
        return _attribute(self.model, self.interface["blocks_path"])

    @property
    def norm(self):
        # A temporary nonregistered collection avoids duplicate state_dict aliases
        # while letting the existing partial-FT API unfreeze every final norm.
        modules = [_attribute(self.model, p) for p in self.interface["norm_paths"]]
        return modules[0] if len(modules) == 1 else nn.ModuleList(modules)

    def _normalize(self, images):
        if not isinstance(images, Tensor) or images.ndim != 4 or tuple(images.shape[1:]) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(f"Expected floating BCHW images (*,{self.in_channels},{self.image_size},{self.image_size})")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("Backbone input must be floating point and finite")
        return (images - self.input_mean.to(dtype=images.dtype)) / self.input_std.to(dtype=images.dtype)

    def _tokens(self, images, indices=None):
        x = self._normalize(images)
        if indices is not None:
            indices = validate_indices(indices, len(images), self.num_patches, device=x.device)
            if not self.supports_masked_tokens:
                raise ValueError("This backbone has no pre-attention masked-context interface; post-attention gathering would leak target patches")
        kind = self.interface["kind"]
        if kind == "native":
            tokens = self.model.forward_tokens(x, indices)
        elif kind == "ijepa":
            tokens = self.model(x, masks=None if indices is None else [indices])
        elif kind == "mae":
            # Official MAE forward_encoder(mask_ratio=0) STILL permutes tokens.
            # Follow its encoder path explicitly, with ordered patches and optional
            # deterministic visible indices selected BEFORE every attention block.
            tokens = self.model.patch_embed(x) + self.model.pos_embed[:, 1:]
            if indices is not None:
                tokens = gather_tokens(tokens, indices)
            cls = (self.model.cls_token + self.model.pos_embed[:, :1]).expand(len(x), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            for block in self.model.blocks:
                tokens = block(tokens)
            tokens = self.model.norm(tokens)
        elif kind == "huggingface":
            tokens = self.model(pixel_values=x, return_dict=True).last_hidden_state
        else:
            kwargs = {}
            if indices is not None:
                kwargs[self.interface["mask_argument"]] = [indices] if self.interface["mask_as_list"] else indices
            tokens = _attribute(self.model, self.interface["method"])(x, **kwargs)
            if self.interface["output_key"] is not None:
                tokens = tokens[self.interface["output_key"]]
        expected_count = (self.num_patches if indices is None else indices.shape[1]) + self.interface["prefix_tokens"]
        if not isinstance(tokens, Tensor) or tuple(tokens.shape) != (len(images), expected_count, self.embed_dim):
            raise ValueError(f"Backbone token contract requires {(len(images), expected_count, self.embed_dim)}, got {getattr(tokens, 'shape', type(tokens))}; check image/patch size and prefix layout")
        return tokens

    def forward_tokens(self, images: Tensor, indices: Tensor | None = None) -> Tensor:
        """Ordered patch tokens only; CLS/register prefix tokens are excluded."""
        return self._tokens(images, indices)[:, self.interface["prefix_tokens"]:]

    def forward(self, images: Tensor) -> Tensor:
        tokens = self._tokens(images)
        if self.spec["pooling"] == "cls":
            result = tokens[:, self.interface["cls_index"]]
        elif self.spec["pooling"] == "model_head":
            result = self.model.forward_head(tokens, pre_logits=True)
        else:
            result = tokens[:, self.interface["prefix_tokens"]:].mean(dim=1)
        if not isinstance(result, Tensor) or tuple(result.shape) != (len(images), self.embed_dim):
            raise ValueError("Backbone pooled output must have shape (batch, embed_dim)")
        return result

    def architecture_metadata(self):
        return {"architecture": self.provenance["architecture_class"], "backend": self.spec["backend"],
                "image_size": self.image_size, "patch_size": self.patch_size, "embed_dim": self.embed_dim,
                "depth": len(self.blocks), "num_register_tokens": self.num_register_tokens,
                "is_paper_pretrained_model": False, "supports_masked_tokens": self.supports_masked_tokens}


def build_backbone(spec: dict) -> BackboneAdapter:
    """Build on CPU from existing local artifacts; optional dependencies are lazy."""
    spec = validate_backbone_spec(spec, check_files=True)
    provenance: dict[str, Any] = {}
    if spec["backend"] == "native":
        model = MiniViT(**spec["kwargs"])
        provenance["architecture_status"] = "project_native_architecture_not_official_pretrained_family"
    elif spec["backend"] == "local_factory":
        model, provenance = _create_local(spec)
    elif spec["backend"] == "timm":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm backend requires an already-installed compatible timm package") from exc
        model = timm.create_model(spec["architecture"], pretrained=False, **spec["kwargs"])
        provenance["timm_version"] = getattr(timm, "__version__", "unknown")
        provenance["pretrained_argument"] = False
    else:
        model, provenance = _create_huggingface(spec)
    if not isinstance(model, nn.Module):
        raise TypeError("Backbone constructor must return torch.nn.Module")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Backbone constructors must initialize on CPU; device placement belongs to the experiment runner")
    if spec.get("checkpoint"):
        provenance.update(_load_external_checkpoint(model, spec["checkpoint"]))
    random_init = spec["backend"] != "huggingface" and not spec.get("checkpoint")
    provenance.update({"source": "explicit_random_initialization" if random_init else "local_checkpoint",
                       "mock": random_init, "no_automatic_download": True})
    return BackboneAdapter(model, spec, provenance)
