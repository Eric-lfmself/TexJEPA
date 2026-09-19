"""Run every TexJEPA component on bounded, synthetic CPU inputs.

This command validates implementation plumbing only. It cannot reproduce the
paper's numerical results or silently start real-data/pretrained-Huge execution.
Example: python -m scripts.run_all --dry-run --profile smoke --output outputs/smoke
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import shutil
import tempfile
import uuid
import subprocess
import sys
import time

import torch
from torch import nn

from configs import load_config, validate_config
from data import DummyCXRDataset, collate_cxr, deterministic_split
from evaluation.lesion_occlusion import evaluate_lesion_occlusion, occlude
from evaluation.mitigation import evaluate_mitigations
from evaluation.robustness import PerturbationSpec, evaluate_robustness
from evaluation.token_diagnostics import compare_token_drift, gradient_activation_saliency
from models import (BlockMaskSampler, IJEPA, MAE, MiniViT, PostTrainingObjective,
                    create_encoder, save_checkpoint, warm_start)
from models.nca import NoiseConsistencyAdapter, train_nca
from perturbations import gaussian_noise
from probing import ProbeConfig, fit_probe, positive_class_weights
from report.build_tables import build_tables, sanitize


EVIDENCE = "synthetic_smoke"


def _band_endpoint(value):
    """Keep paper labels familiar without rounding distinct configured bounds."""
    value = float(value)
    if value == 0:
        value = 0.0
    return f"{value:.2f}" if value in (0.0, 0.2, 0.45, 1.0) else repr(value)


def configured_perturbations(config):
    """Expand every configured severity and band, preserving explicit parameters."""
    p = config["perturbations"]
    specs = []
    for name, key, parameter in (("gaussian_noise", "gaussian_noise", "sigma"),
                                  ("gaussian_blur", "gaussian_blur", "sigma"),
                                  ("brightness_shift", "brightness_shift", "shift"),
                                  ("contrast_scale", "contrast_scaling", "factor")):
        specs.extend(PerturbationSpec(name, float(value), {parameter: float(value)}) for value in p[key])
    normalization = p["frequency_normalization"]
    specs.append(PerturbationSpec("low_pass", p["low_pass_cutoff"],
                                  {"cutoff": p["low_pass_cutoff"], "normalization": normalization}))
    for mode in p["band_modes"]:
        if mode not in ("stop", "pass"):
            raise ValueError("Configured frequency band_modes must be stop/pass")
        for low, high in p["frequency_bands"]:
            specs.append(PerturbationSpec(f"band_{mode}", f"{_band_endpoint(low)}-{_band_endpoint(high)}",
                                          {"low": low, "high": high, "normalization": normalization}))
    return specs


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return sanitize(value)


def _versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in ("torch", "torchvision", "numpy", "pillow", "pytest"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not_installed"
    return versions


class _AdaptedEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.nca = model

    def forward(self, images):
        return self.nca.encode(images)


_CHECKPOINT_FILES = frozenset({
    "ijepa_main_synthetic.pt", "mae_main_synthetic.pt", "v31_lineage_fixture.pt",
    "v4_lineage_fixture.pt", "v5_lineage_fixture.pt", "v6_lineage_fixture.pt",
})
_TABLE_FILES = frozenset({"manifest.json", "INDEX.md", "robustness_long.csv", "robustness_long.md"} | {
    f"table_{number}.{suffix}" for number in ("IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII")
    for suffix in ("csv", "md")
})
_MANAGED_DIRS = {"checkpoints": _CHECKPOINT_FILES, "tables": _TABLE_FILES}


def _published_paths(value, staging, destination):
    """Remap only our temporary absolute path prefix, including checkpoint lineage."""
    if isinstance(value, dict):
        return {key: _published_paths(item, staging, destination) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_published_paths(item, staging, destination) for item in value]
    if isinstance(value, str):
        prefix = str(staging)
        if value == prefix or value.startswith(prefix + os.sep):
            return str(destination) + value[len(prefix):]
    return value


def _output_extras(destination):
    """Recognize existing generated files and preserve every other entry by rename.

    Never descend into caller-owned subtrees or follow output symlinks. Existing
    files using generated names require the provenance of a completed smoke run.
    """
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Pipeline output must be a directory, not a file or symlink")
    if not destination.exists():
        return []
    owned = []
    extras = []
    for entry in destination.iterdir():
        if entry.name == "results.json":
            owned.append(entry)
        elif entry.name in _MANAGED_DIRS:
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError(f"Managed output directory cannot be a file or symlink: {entry}")
            for child in entry.iterdir():
                if child.name in _MANAGED_DIRS[entry.name]:
                    owned.append(child)
                else:
                    extras.append(child.relative_to(destination))
        else:
            extras.append(entry.relative_to(destination))
    if any(entry.is_symlink() or not entry.is_file() for entry in owned):
        raise ValueError("Managed output files must be regular files, not directories or symlinks")
    if owned:
        result_path = destination / "results.json"
        try:
            if not result_path.is_file() or result_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Missing or oversized prior result")
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            metadata = previous["metadata"]
            recognized = (metadata["evidence"] == EVIDENCE
                          and metadata["data_source"] == "DummyCXRDataset"
                          and metadata["real_experiments_run"] is False
                          and isinstance(metadata["config"], dict))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Refusing to replace generated filenames without a valid prior smoke result") from exc
        if not recognized:
            raise ValueError("Refusing to overwrite output belonging to another dataset or tool")
    return extras


def _remove_old_generated(backup):
    """Remove known old artifacts only; unexpected entries keep their backup."""
    for relative in [Path("results.json"), *(Path(folder) / name
                     for folder, names in _MANAGED_DIRS.items() for name in names)]:
        try:
            (backup / relative).unlink(missing_ok=True)
        except OSError:
            pass
    for folder in (*_MANAGED_DIRS, "."):
        try:
            (backup / folder).rmdir()
        except OSError:
            pass


def _publish_run(staging, destination):
    """Publish by same-filesystem renames, restoring the prior run on failure.

    Readers may briefly see an absent directory between renames; a run is never
    published with some checkpoints or tables from its predecessor. This does
    not support concurrent writers to one output directory.
    """
    backup = None
    transferred = []
    try:
        extras = _output_extras(destination)
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.previous-{uuid.uuid4().hex}")
            os.replace(destination, backup)
            for relative in extras:
                os.replace(backup / relative, staging / relative)
                transferred.append(relative)
        os.replace(staging, destination)
    except BaseException:
        try:
            if backup is not None and backup.exists():
                for relative in reversed(transferred):
                    os.replace(staging / relative, backup / relative)
                os.replace(backup, destination)
        except BaseException as rollback_error:
            # Unknown caller files may still be in staging: never delete it.
            raise RuntimeError(f"Output rollback failed; preserved files at {backup} and {staging}") from rollback_error
        shutil.rmtree(staging)
        raise
    if backup is not None:
        _remove_old_generated(backup)


def run_pipeline(config, output_dir):
    """Run a bounded smoke measurement and publish one consistent artifact set.

    Fitting uses only training images. All artifacts are prepared in a sibling
    directory; a failed run preserves existing results and caller-owned files.
    """
    validate_config(config, for_execution=True)
    # Resolve the caller's directory alias once and publish into its real
    # target, leaving the alias itself untouched.
    destination = Path(output_dir).resolve()
    source_root = Path(__file__).resolve().parents[1]
    current = Path.cwd().resolve()
    if destination in (source_root, current, *source_root.parents, *current.parents):
        raise ValueError("Use a dedicated output directory outside the source root and its parents")
    _output_extras(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        results = _run_pipeline(copy.deepcopy(config), staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    _publish_run(staging, destination)
    return results


def _run_pipeline(config, output_dir, published_output_dir):
    """Build a complete run in an unpublished directory."""
    specs = configured_perturbations(config)
    runtime, model_config = config["runtime"], config["model"]
    torch.manual_seed(runtime["seed"])
    torch.set_num_threads(runtime["num_threads"])
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    seed = runtime["seed"]
    dataset = DummyCXRDataset(size=config["data"]["smoke_samples"], image_size=model_config["image_size"], seed=seed)
    train_count = len(dataset) // 2
    train_indices, test_indices = deterministic_split(dataset.image_ids, train_count, len(dataset)-train_count,
                                                       seed=config["data"]["split_seed"])
    train_samples, test_samples = [dataset[i] for i in train_indices], [dataset[i] for i in test_indices]
    # Keep the final training batch >=2 for BN without dropping a sample.
    def batches(samples):
        chunks = [samples[i:i+runtime["batch_size"]] for i in range(0, len(samples), runtime["batch_size"])]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        return [collate_cxr(chunk) for chunk in chunks]
    train_batches, test_batches = batches(train_samples), batches(test_samples)
    results = {"metadata": {"evidence": EVIDENCE, "real_experiments_run": False,
                             "paper_metrics_reproduced": False, "data_source": "DummyCXRDataset",
                             "model_source": "explicit random MiniViT initialization",
                             "downloads_performed": False, "device": "cpu", "seed": seed,
                             "config": config,
                             "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                             "versions": _versions(), "train_image_ids": [s["image_id"] for s in train_samples],
                             "test_image_ids": [s["image_id"] for s in test_samples],
                             "class_names": [f"synthetic_class_{i:02d}" for i in range(15)],
                             "class_mapping_is_synthetic": True,
                             "lineage_epoch_labels_are_fixture_only": True,
                             "checkpoint_note": "201/50/300/95/97 are paper lineage labels, never actual smoke epochs",
                             "source_paper": "texjepa-manuscript (not distributed)",
                             "perturbation_count": len(specs), "synthetic_checkpoints": []},
               "robustness": [], "lesion": [], "lesion_details": [], "training": [],
               "post_training": [], "mitigation": [], "nca": [], "train_aug": [], "token_diagnostics": [], "interventions": []}
    model_keys = ("image_size", "patch_size", "in_channels", "embed_dim", "depth", "num_heads")
    kwargs = {key: model_config[key] for key in model_keys}
    def new_encoder(registers=0):
        return MiniViT(**kwargs, num_register_tokens=registers).cpu()
    def new_jepa(registers=0):
        return IJEPA(new_encoder(registers), predictor_dim=model_config["predictor_dim"],
                      predictor_depth=model_config["predictor_depth"], num_heads=model_config["num_heads"]).cpu()
    def fixture_metadata(variant, epoch, steps, **extra):
        return {"variant": variant, "epoch": epoch, "epoch_is_lineage_label": True,
                "actual_training_steps": steps, "mock": True, "evidence": EVIDENCE, **extra}
    def checkpoint(filename, model, metadata):
        metadata = _published_paths(metadata, output_dir, published_output_dir)
        path = save_checkpoint(checkpoint_dir / filename, model, metadata)
        results["metadata"]["synthetic_checkpoints"].append({
            "path": str(published_output_dir / "checkpoints" / filename), "metadata": metadata})
        return path
    def add_training(phase, model_name, record):
        results["training"].append({"phase": phase, "model": model_name, "evidence": EVIDENCE, **record})
    def mask_sampler(mode="baseline"):
        p = config["post_training"]
        registers = mode in ("register", "v5")
        return BlockMaskSampler(model_config["image_size"] // model_config["patch_size"], variant=mode, seed=seed,
                                context_scale=p["register_context_scale"] if registers else p["context_scale"],
                                target_scale=p["target_scale"],
                                num_target_blocks=p["register_target_blocks"] if registers else p["target_blocks"],
                                target_aspect_ratio=p["register_target_aspect"] if registers else p["target_aspect"])
    def pretrain_once(model, kind):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        images = train_batches[0]["image"]
        if kind == "ijepa":
            masks = mask_sampler()(len(images))
            measured = model(images, masks["context_indices"], masks["target_indices"])
        else:
            masks = None
            measured = model(images, mask_ratio=.75, generator=torch.Generator().manual_seed(seed))
        optimizer.zero_grad(set_to_none=True)
        measured["loss"].backward()
        optimizer.step()
        if kind == "ijepa":
            model.update_ema(.996)
        add_training("synthetic_pretraining", kind, {"actual_steps": 1, "loss": measured["loss"].item(),
                                                     "mask_metadata": None if masks is None else masks["metadata"]})
    main_jepa = new_jepa()
    pretrain_once(main_jepa, "ijepa")
    main_mae = MAE(new_encoder(), decoder_dim=model_config["predictor_dim"],
                    decoder_depth=model_config["predictor_depth"], num_heads=model_config["num_heads"])
    pretrain_once(main_mae, "mae")
    checkpoint("ijepa_main_synthetic.pt", main_jepa, fixture_metadata("main_ijepa", 300, 1))
    checkpoint("mae_main_synthetic.pt", main_mae, fixture_metadata("main_mae", 300, 1))

    def fit(encoder, paper_model, protocol, group, evaluate=True, lesions=False):
        name = f"{paper_model} [minirandom]"
        p = config["probing"]
        probe_config = ProbeConfig(protocol=protocol, epochs=p["smoke_epochs"], num_classes=15,
                                    hidden_dim=p["hidden_dim"], dropout=p["dropout"], head_lr=p["head_lr"],
                                    encoder_lr=p["encoder_lr"], momentum=p["momentum"], weight_decay=p["weight_decay"])
        fitted, training = fit_probe(copy.deepcopy(encoder), train_batches, probe_config)
        add_training("probing", name, training)
        rows = []
        if evaluate:
            rows = evaluate_robustness(fitted.encoder, fitted.head, test_batches, specs, model_name=name,
                                        protocol=protocol, seed=seed, evidence=EVIDENCE)
            for row in rows:
                row.update(group=group, paper_model=paper_model)
            results["robustness"].extend(rows)
        if lesions:
            lesion_config = config["lesion"]
            if lesion_config["confidence"] != .95:
                raise ValueError("The current lesion bootstrap implements confidence=0.95 only")
            detail = evaluate_lesion_occlusion(fitted.encoder, fitted.head, test_samples, model_name=name,
                                               protocol=protocol, fill=lesion_config["fill"], controls=lesion_config["controls"],
                                               bootstrap_iterations=lesion_config["bootstrap_replicates"],
                                               seed=lesion_config["seed"], evidence=EVIDENCE)
            summary = {**detail["overall"], "n_skipped": detail["overall"]["n_skipped_records"],
                       "group": group, "paper_model": paper_model}
            results["lesion"].append(summary)
            results["lesion_details"].append({"model": name, "protocol": protocol, "evidence": EVIDENCE,
                                                "group": group, "paper_model": paper_model, **detail})
        return fitted, rows

    main_linear = {}
    for paper_name, encoder in (("I-JEPA-H/300", main_jepa.context_encoder), ("MAE-H/300", main_mae.encoder)):
        for protocol in ("linear", "mlp", "partial_ft"):
            fitted, rows = fit(encoder, paper_name, protocol, "main", lesions=protocol == "linear")
            if protocol == "linear":
                main_linear[paper_name] = (fitted, rows)

    v31 = copy.deepcopy(main_jepa)
    p31 = checkpoint("v31_lineage_fixture.pt", v31, fixture_metadata("v3.1", 201, 1,
                     source="copy_of_synthetic_main_jepa_fixture"))
    fit(v31.context_encoder, "v3.1 (Baseline)", "linear", "post", lesions=True)
    v4_path = None
    for variant, mode, registers in (("v4", "noise", 0), ("v5", "register", 4), ("v6", "vicreg", 0)):
        model = new_jepa(registers)
        lineage = warm_start(p31 if variant == "v4" else v4_path, model, variant)
        p = config["post_training"]
        objective = PostTrainingObjective(mode, gaussian_sigma=p["gaussian_sigma"], poisson_peak=p["poisson_rate"],
                                           jpeg_quality=p["jpeg_quality"], lambda_var=p["lambda_var"], lambda_cov=p["lambda_cov"])
        images = train_batches[0]["image"]
        masks = mask_sampler(mode)(len(images))
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.001)
        measured = objective(model, images, masks["context_indices"], masks["target_indices"],
                              generator=torch.Generator().manual_seed(seed + int(variant[1:])))
        optimizer.zero_grad(set_to_none=True)
        measured["loss"].backward()
        optimizer.step()
        model.update_ema(.996)
        path = checkpoint(f"{variant}_lineage_fixture.pt", model,
                           fixture_metadata(variant, 50 if variant == "v4" else 1, 1,
                                              warm_start=lineage, actual_step_scope="new_post_training_branch"))
        if variant == "v4":
            v4_path = path
        results["post_training"].append({"model": f"{variant} [minirandom]", "evidence": EVIDENCE,
                                          "mode": mode, "actual_steps": 1, "warm_start": lineage,
                                          "loss": measured["loss"].item(),
                                          "asymmetric_loss": measured["asymmetric_loss"].item(),
                                          "variance_loss": measured["variance_loss"].item(),
                                          "covariance_loss": measured["covariance_loss"].item(),
                                          "noise_kind": measured["noise_kind"], "mask_metadata": masks["metadata"]})
        fit(model.context_encoder, f"{variant} (+{mode})", "linear", "post", lesions=True)

    for family, paper_name in (("eva_x", "EVA-X-B/300"), ("rad_dino", "RAD-DINO-B/300")):
        encoder = create_encoder(family, use_random_init=True, **kwargs)
        results["metadata"].setdefault("external_models", []).append({"paper_model": paper_name, **encoder.provenance})
        fit(encoder, paper_name, "linear", "external", lesions=True)
    for paper_name in ("I-JEPA-H/95", "I-JEPA-H/201", "MAE-H/97"):
        fit(new_encoder(), paper_name, "linear", "historical", evaluate=False, lesions=True)

    def condition(rows, name, sigma=None):
        return next((row for row in rows if row["perturbation"] == name and
                     (sigma is None or float(row["severity"]) == sigma)), None)
    def intervention_rows(name, method, records, original):
        clean = condition(records, "clean")
        rows = []
        for noisy in records:
            if noisy["perturbation"] != "gaussian_noise":
                continue
            sigma = float(noisy["severity"])
            raw = condition(original, "gaussian_noise", sigma)
            rows.append({"model": name, "intervention": method,
                         "protocol": noisy["protocol"], "evidence": EVIDENCE,
                         "clean_auroc": None if clean is None else clean["auroc"],
                         "noise_auroc": noisy["auroc"],
                         "raw_drift": None if raw is None else raw["drift"],
                         "adapted_drift": noisy["drift"], "noise_sigma": sigma})
        return rows
    for paper_name, (fitted, original_rows) in main_linear.items():
        name = f"{paper_name} [minirandom]"
        # The unmodified branch has already been measured with these inputs,
        # seed and fitted probe. Copy it before adding mitigation metadata.
        mitigation = [{**{key: value for key, value in copy.deepcopy(row).items()
                           if key not in ("group", "paper_model")},
                       "mitigation": "none", "mitigation_parameters": {}} for row in original_rows]
        mitigation.extend(evaluate_mitigations(fitted.encoder, fitted.head, test_batches, specs,
                                               model_name=name, protocol="linear", seed=seed, evidence=EVIDENCE,
                                               include_unmodified=False))
        results["mitigation"].extend(mitigation)
        for method in ("none", "median", "gaussian"):
            subset = [row for row in mitigation if row["mitigation"] == method]
            results["interventions"].extend(intervention_rows(name, f"input_{method}", subset, original_rows))
        # Fig.8 train_aug is a separate trained probe with the same frozen
        # encoder and split. The manuscript does not report its augmentation
        # distribution; Gaussian sigma=.05 is an explicit implementation choice.
        probe = config["probing"]
        aug_config = ProbeConfig(protocol="linear", epochs=probe["smoke_epochs"], num_classes=15,
                                  hidden_dim=probe["hidden_dim"], dropout=probe["dropout"],
                                  head_lr=probe["head_lr"], encoder_lr=probe["encoder_lr"],
                                  momentum=probe["momentum"], weight_decay=probe["weight_decay"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            augmented_probe, aug_training = fit_probe(
                copy.deepcopy(fitted.encoder), train_batches, aug_config,
                augmentation=lambda images, *, generator: gaussian_noise(images, .05, generator=generator),
                augmentation_seed=seed, device="cpu")
        add_training("train_aug", name, {**aug_training,
                     "augmentation": {"kind": "gaussian_noise", "sigma": .05,
                                      "seed": seed, "historical_parameter_recovered": False}})
        aug_rows = evaluate_robustness(augmented_probe.encoder, augmented_probe.head, test_batches, specs,
                                        model_name=name, protocol="linear_train_aug", seed=seed, evidence=EVIDENCE)
        results["train_aug"].append({"model": name, "protocol": "linear_train_aug", "evidence": EVIDENCE,
                                     "robustness": aug_rows, "training": aug_training})
        results["interventions"].extend(intervention_rows(name, "train_aug", aug_rows, original_rows))
        nca_config = config["nca"]
        nca = NoiseConsistencyAdapter(copy.deepcopy(fitted.encoder), model_config["embed_dim"],
                                       num_classes=15, hidden_dim=nca_config["hidden_dim"], head=copy.deepcopy(fitted.head))
        weights, undefined = positive_class_weights(torch.stack([sample["labels"] for sample in train_samples]))
        nca_history = train_nca(nca, train_batches, epochs=1, learning_rate=nca_config["learning_rate"],
                                 alignment_weight=nca_config["feature_weight"],
                                 consistency_weight=nca_config["prediction_weight"],
                                 supervised_weight=nca_config["supervised_weight"], pos_weight=weights,
                                 seed=seed, max_steps=2)
        nca_rows = evaluate_robustness(_AdaptedEncoder(nca), nca.head, test_batches, specs,
                                       model_name=name, protocol="nca", seed=seed, evidence=EVIDENCE)
        results["nca"].append({"model": name, "protocol": "nca", "evidence": EVIDENCE,
                               "history": nca_history, "robustness": nca_rows,
                               "undefined_prevalence_classes": undefined, "weight_source": "train_split_only"})
        results["interventions"].extend(intervention_rows(name, "noise_consistency_adapter", nca_rows, original_rows))
        # Use an annotated held-out image for both local occlusion and global
        # noise maps; class-specific attribution remains only a weak diagnostic.
        sample = next((sample for sample in test_samples if sample["boxes"]), None)
        if sample is None:
            results["token_diagnostics"].append({"model": name, "evidence": EVIDENCE,
                                                "status": "no_annotated_held_out_image"})
        else:
            images = sample["image"].unsqueeze(0)
            noisy = gaussian_noise(images, .05, generator=torch.Generator().manual_seed(seed))
            box = sample["boxes"][0]
            occluded = occlude(sample["image"], box["bbox"], fill=config["lesion"]["fill"]).unsqueeze(0)
            token_record = {"model": name, "protocol": "linear", "evidence": EVIDENCE,
                            "image_id": sample["image_id"], "class_index": box["class_index"],
                            "noise_sigma": .05, "lesion_bbox": box["bbox"],
                            "noise": compare_token_drift(fitted.encoder, images, noisy),
                            "lesion_occlusion": compare_token_drift(fitted.encoder, images, occluded),
                            "saliency": gradient_activation_saliency(fitted.encoder, fitted.head, images, box["class_index"])}
            results["token_diagnostics"].append(token_record)

    results["metadata"]["elapsed_seconds"] = time.monotonic() - started
    results = _jsonable(_published_paths(results, output_dir, published_output_dir))
    destination = output_dir / "results.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    build_tables(results, output_dir / "tables")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="Required acknowledgement: run bounded synthetic CPU measurements only")
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    tests = parser.add_mutually_exclusive_group()
    tests.add_argument("--run-tests", action="store_true",
                       help="Run the source checkout's test suite before the synthetic pipeline (requires the test extra)")
    tests.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_config(args.profile)
    validate_config(config, for_execution=True)
    if args.run_tests:
        source_root = Path(__file__).resolve().parents[1]
        if not (source_root / "tests").is_dir() or not (source_root / "pyproject.toml").is_file():
            parser.error("--run-tests requires a source checkout installed with the test extra")
        subprocess.run([sys.executable, "-m", "pytest", "tests", "-o", "addopts=", "-q"],
                       cwd=source_root, check=True, timeout=180)
    results = run_pipeline(config, args.output)
    print(json.dumps({"evidence": EVIDENCE, "real_experiments_run": False,
                      "results": str((args.output / "results.json").resolve()),
                      "tables": str((args.output / "tables" / "INDEX.md").resolve()),
                      "robustness_records": len(results["robustness"]),
                      "lesion_models": len(results["lesion"]),
                      "elapsed_seconds": results["metadata"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
