"""Configuration separates paper-scale descriptions from bounded CPU execution.

Unspecified paper details are implementation assumptions, listed in
``docs/METHODS.md``. Loading a paper config never creates a model.
"""
import copy
import json
import math
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def load_config(profile="smoke", overrides=None):
    if profile not in {"paper", "smoke"}:
        raise ValueError("profile must be paper or smoke")
    config = json.loads((CONFIG_DIR / "paper.json").read_text())
    if profile == "smoke":
        _merge(config, json.loads((CONFIG_DIR / "smoke.json").read_text()))
    if overrides:
        _merge(config, copy.deepcopy(overrides))
    validate_config(config)
    return config


def _merge(base, updates):
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(value, dict) and isinstance(base[key], dict):
            _merge(base[key], value)
        else:
            base[key] = value


def _positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(name, value, *, minimum=None, maximum=None, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    if ((positive and value <= 0) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} is outside its supported range")


def _bounds(name, values, *, scale=False):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain two bounds")
    for value in values:
        _finite_number(name, value, positive=True, maximum=1 if scale else None)
    if values[0] > values[1]:
        raise ValueError(f"{name} bounds must be increasing")


def _validate_workload(config, for_execution):
    """Validate consumed parameters before any smoke allocation or fitting."""
    counts = {
        "runtime": ("batch_size", "num_threads"),
        "data": ("num_classes", "train_size", "test_size", "smoke_samples"),
        "probing": ("linear_epochs", "mlp_epochs", "partial_ft_epochs", "smoke_epochs", "hidden_dim"),
        "lesion": ("controls", "bootstrap_replicates"),
        "post_training": ("register_tokens", "target_blocks", "register_target_blocks", "v5_v6_parent_epoch"),
        "nca": ("hidden_dim",),
    }
    for section, keys in counts.items():
        for key in keys:
            _positive_integer(f"{section}.{key}", config[section][key])
    probe, lesion, post, nca = (config[key] for key in ("probing", "lesion", "post_training", "nca"))
    for key in ("head_lr", "encoder_lr"):
        _finite_number(f"probing.{key}", probe[key], positive=True)
    if (for_execution or probe["protocol"] == "partial_ft") and probe["encoder_lr"] >= probe["head_lr"]:
        raise ValueError("Partial FT encoder learning rate must be lower than head LR")
    _finite_number("probing.dropout", probe["dropout"], minimum=0)
    if probe["dropout"] >= 1:
        raise ValueError("probing.dropout must be less than one")
    for key in ("momentum", "weight_decay"):
        _finite_number(f"probing.{key}", probe[key], minimum=0)
    _finite_number("lesion.fill", lesion["fill"], minimum=0, maximum=1)
    _finite_number("lesion.confidence", lesion["confidence"], positive=True, maximum=1)
    for key in ("context_scale", "register_context_scale", "target_scale"):
        _bounds(f"post_training.{key}", post[key], scale=True)
    for key in ("target_aspect", "register_target_aspect"):
        _bounds(f"post_training.{key}", post[key])
    for key in ("lambda_var", "lambda_cov", "gaussian_sigma"):
        _finite_number(f"post_training.{key}", post[key], minimum=0)
    _finite_number("post_training.poisson_rate", post["poisson_rate"], positive=True)
    _positive_integer("post_training.jpeg_quality", post["jpeg_quality"])
    if post["jpeg_quality"] > 100:
        raise ValueError("post_training.jpeg_quality must be in [1,100]")
    _finite_number("nca.learning_rate", nca["learning_rate"], positive=True)
    for key in ("feature_weight", "prediction_weight", "supervised_weight"):
        _finite_number(f"nca.{key}", nca[key], minimum=0)
    if not any(nca[key] > 0 for key in ("feature_weight", "prediction_weight", "supervised_weight")):
        raise ValueError("At least one NCA loss weight must be positive")

    perturb = config["perturbations"]
    scalar_lists = ("gaussian_noise", "gaussian_blur", "brightness_shift", "contrast_scaling")
    for key in (*scalar_lists, "frequency_bands", "band_modes"):
        if not isinstance(perturb[key], (list, tuple)):
            raise ValueError(f"perturbations.{key} must be a list")
    # One low-pass condition is always included; the clean reference is not a corruption.
    conditions = sum(len(perturb[key]) for key in scalar_lists) + 1
    conditions += len(perturb["frequency_bands"]) * len(perturb["band_modes"])
    if for_execution and (conditions > 32 or any(
            len(perturb[key]) > 32 for key in (*scalar_lists, "frequency_bands", "band_modes"))):
        raise ValueError("Smoke runner permits at most 32 perturbation conditions and entries per list")
    for key in scalar_lists:
        for value in perturb[key]:
            _finite_number(f"perturbations.{key}", value,
                           minimum=None if key == "brightness_shift" else 0)
        if len(set(map(float, perturb[key]))) != len(perturb[key]):
            raise ValueError(f"Duplicate perturbation condition in {key}")
    _finite_number("perturbations.low_pass_cutoff", perturb["low_pass_cutoff"], positive=True, maximum=1)
    if perturb["frequency_normalization"] not in ("corner_nyquist", "axis_nyquist"):
        raise ValueError("Unknown frequency normalization")
    if any(mode not in ("stop", "pass") for mode in perturb["band_modes"]):
        raise ValueError("Configured frequency band_modes must be stop/pass")
    if len(set(perturb["band_modes"])) != len(perturb["band_modes"]):
        raise ValueError("Duplicate frequency band mode")
    bands = []
    for band in perturb["frequency_bands"]:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("Frequency bands must contain two bounds")
        for value in band:
            _finite_number("Frequency band endpoint", value, minimum=0, maximum=1)
        low, high = map(float, band)
        if low >= high:
            raise ValueError("Frequency bands require low < high")
        bands.append((low, high))
    if len(set(bands)) != len(bands):
        raise ValueError("Duplicate frequency band condition")
    if for_execution:
        if lesion["controls"] > 5 or lesion["bootstrap_replicates"] > 2000:
            raise ValueError("Smoke lesion workload is limited to 5 controls and 2000 bootstrap replicates")
        if post["target_blocks"] > 6 or post["register_target_blocks"] > 6:
            raise ValueError("Smoke masking is limited to 6 target blocks")
        if any(sigma > 4 for sigma in perturb["gaussian_blur"]):
            raise ValueError("Smoke Gaussian blur sigma must be at most 4")
        if lesion["confidence"] != .95:
            raise ValueError("The current lesion bootstrap implements confidence=0.95 only")


def validate_config(config, for_execution=False):
    runtime, model = config["runtime"], config["model"]
    if runtime["device"] != "cpu":
        raise ValueError("Only explicit CPU execution is supported")
    if config["probing"]["protocol"] not in ("linear", "mlp", "partial_ft"):
        raise ValueError("Unsupported probing protocol")
    if config["post_training"]["mode"] not in ("baseline", "noise", "register", "vicreg"):
        raise ValueError("Unsupported post-training mode")
    for key in ("image_size", "patch_size", "in_channels", "embed_dim", "depth", "num_heads", "predictor_dim", "predictor_depth"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"Model {key} must be a positive integer")
    if type(model["num_register_tokens"]) is not int or model["num_register_tokens"] < 0:
        raise ValueError("Register count must be a nonnegative integer")
    if model["image_size"] % model["patch_size"] or model["embed_dim"] % model["num_heads"] or model["predictor_dim"] % model["num_heads"]:
        raise ValueError("Model image/patch and embedding/head dimensions must divide")
    if config["data"]["num_classes"] != 15:
        raise ValueError("Paper protocol specifies 15 classes")
    _validate_workload(config, for_execution)
    if for_execution:
        if not runtime["dry_run"] or not runtime["use_random_init"]:
            raise ValueError("Bundled runner executes synthetic smoke checks only")
        if config["data"]["source"] != "dummy" or not config["model"]["mini"]:
            raise ValueError("Smoke execution requires dummy data and mini ViT")
        if not (2 <= runtime["batch_size"] <= 4 and model["image_size"] in (32, 64)):
            raise ValueError("Smoke input must be batch 2–4 and image size 32 or 64")
        if (model["embed_dim"] > 64 or model["predictor_dim"] > 64 or model["depth"] > 2
                or model["predictor_depth"] > 2 or model["num_register_tokens"] > 4
                or model["in_channels"] != 3 or (model["image_size"] // model["patch_size"])**2 > 64):
            raise ValueError("Smoke runner refuses oversized models")
        if model["depth"] < 2 or (model["image_size"] // model["patch_size"]) < 2:
            raise ValueError("Complete smoke suite requires two blocks and at least a 2x2 patch grid")
        if not (4 <= config["data"]["smoke_samples"] <= 8):
            raise ValueError("Smoke runner permits 4–8 synthetic examples")
        if not (1 <= runtime["num_threads"] <= 4 and 1 <= config["probing"]["hidden_dim"] <= 512
                and 1 <= config["nca"]["hidden_dim"] <= 64):
            raise ValueError("Smoke threads and auxiliary heads must remain small")
        if config["probing"]["smoke_epochs"] != 1:
            raise ValueError("Smoke runner performs one synthetic epoch only")
    return config
