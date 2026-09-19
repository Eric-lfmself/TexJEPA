"""Frozen paired robustness evaluation, TexJEPA Sec. IV-B/C, Eqs. 1-2.

Each clean image and its corruptions use the same encoder/head in eval mode,
under no_grad. The caller controls image normalization inside the encoder.
Determinism assumes the same sample order/batching; every perturbation gets
an independent seeded generator. CSV-ready rows include the evidence label.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import math

import torch

import perturbations as perturb
from metrics import cosine_drift, delta_auroc, multilabel_auroc


@contextmanager
def evaluation_mode(*models):
    states = {module: module.training for model in models for module in model.modules()}
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in states.items():
            module.training = training


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    severity: float | str
    parameters: dict = field(default_factory=dict)

    def apply(self, images, *, generator):
        functions = {"gaussian_noise": perturb.gaussian_noise,
                     "gaussian_blur": perturb.gaussian_blur,
                     "brightness_shift": perturb.brightness_shift,
                     "contrast_scale": perturb.contrast_scale,
                     "low_pass": perturb.low_pass,
                     "band_pass": perturb.band_pass,
                     "band_stop": perturb.band_stop}
        if self.name not in functions:
            raise ValueError(f"Unknown perturbation {self.name!r}")
        return functions[self.name](images, generator=generator, **self.parameters)


def paper_perturbations(*, normalization="corner_nyquist"):
    """Paper noise/bands plus configurable Fig. 4 implementation defaults."""
    specs = [PerturbationSpec("gaussian_noise", sigma, {"sigma": sigma})
             for sigma in (0.05, 0.10, 0.20, 0.30)]
    specs += [PerturbationSpec("gaussian_blur", 1.0, {"sigma": 1.0}),
              PerturbationSpec("brightness_shift", 0.1, {"shift": 0.1}),
              PerturbationSpec("contrast_scale", 0.8, {"factor": 0.8}),
              PerturbationSpec("low_pass", 0.15, {"cutoff": 0.15, "normalization": normalization})]
    specs += [PerturbationSpec(mode, f"{low:.2f}-{high:.2f}",
                               {"low": low, "high": high, "normalization": normalization})
              for mode in ("band_pass", "band_stop")
              for low, high in ((0, 0.2), (0.2, 0.45), (0.45, 1.0))]
    return specs


def evaluate_robustness(encoder, head, batches, perturbations, *, model_name,
                        protocol="linear", seed=42, evidence="synthetic_smoke",
                        preprocess=None):
    """Return table rows including a clean row. Batches have image and labels.

    Optional preprocessing is applied equally to clean and perturbed inputs,
    enabling fair paired mitigation comparisons (Sec. V-I). A preprocessing
    function that expects [0,1] must not silently clip FFT-filtered inputs.

    AUROC values retain their independently valid observations. Delta is
    defined only when the clean and perturbed finite label-and-score masks
    match exactly and both macro AUROCs are defined. Otherwise it is NaN,
    with delta_status explaining the incomparable or undefined result.
    """
    return _evaluate_preprocessors(
        encoder, head, batches, perturbations, [preprocess], model_name=model_name,
        protocol=protocol, seed=seed, evidence=evidence)[0]


def _evaluate_preprocessors(encoder, head, batches, perturbations, preprocessors, *,
                            model_name, protocol="linear", seed=42,
                            evidence="synthetic_smoke"):
    """Stream shared clean/corrupted batches through each preprocessing method.

    Return one row collection per method. Images and random corruptions are
    materialized only for the current batch; score/drift accumulators retain
    the dataset-level evidence needed for AUROC. A one-shot batch stream is
    sufficient, including streams whose order or images vary on later passes.
    """
    specs = list(perturbations)
    transforms = list(preprocessors)
    if not transforms:
        raise ValueError("At least one preprocessing method is required")
    keys = [(spec.name, str(spec.severity)) for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Perturbation name/severity pairs must be unique")
    targets = []
    accumulators = [{"clean_scores": [], "clean_drifts": [],
                     "corrupted_scores": [[] for _ in specs],
                     "corrupted_drifts": [[] for _ in specs]}
                    for _ in transforms]
    generators = {}
    with evaluation_mode(encoder, head):
        for batch in batches:
            images, labels = batch["image"], batch["labels"]
            if images.ndim != 4 or labels.ndim != 2 or images.shape[0] != labels.shape[0]:
                raise ValueError("Expected BCHW images paired with BxC labels")
            # CPU producers may refill the same label buffer for the next
            # batch. Keep the labels observed alongside this batch's scores.
            targets.append(labels.detach().cpu().clone())
            clean_features = []
            for transform, values in zip(transforms, accumulators):
                features = encoder(transform(images) if transform else images)
                if not isinstance(features, torch.Tensor) or features.ndim != 2:
                    raise ValueError("Encoder must return pooled BxD image features")
                logits = head(features)
                if logits.shape != labels.shape:
                    raise ValueError("Head logits must match label shape")
                clean_features.append(features)
                values["clean_scores"].append(logits.detach().cpu())
                values["clean_drifts"].append(cosine_drift(features, features).cpu())
            for index, spec in enumerate(specs):
                if index not in generators:
                    offset = int.from_bytes(hashlib.sha256(repr(keys[index]).encode()).digest()[:4], "big")
                    generators[index] = torch.Generator(device=images.device).manual_seed(seed + offset)
                altered = spec.apply(images, generator=generators[index])
                for transform, values, clean in zip(transforms, accumulators, clean_features):
                    features = encoder(transform(altered) if transform else altered)
                    values["corrupted_scores"][index].append(head(features).detach().cpu())
                    values["corrupted_drifts"][index].append(cosine_drift(clean, features).cpu())
    if not targets:
        raise ValueError("Evaluation received no batches")
    labels = torch.cat(targets)
    return [_robustness_rows(labels, specs, model_name=model_name, protocol=protocol,
                             evidence=evidence, **values) for values in accumulators]


def _robustness_rows(labels, specs, *, clean_scores, clean_drifts,
                     corrupted_scores, corrupted_drifts, model_name, protocol, evidence):
    """Summarize one method using matched clean/perturbed validity masks."""
    clean_score_tensor = torch.cat(clean_scores)
    clean_metric = multilabel_auroc(labels, clean_score_tensor)
    finite_labels = torch.isfinite(labels)
    clean_support = finite_labels & torch.isfinite(clean_score_tensor)
    rows = []
    all_entries = [("clean", 0.0, clean_scores, clean_drifts, {})]
    all_entries += [(spec.name, spec.severity, corrupted_scores[i], corrupted_drifts[i], spec.parameters)
                    for i, spec in enumerate(specs)]
    for name, severity, scores, drifts, parameters in all_entries:
        score_tensor = torch.cat(scores)
        metric = multilabel_auroc(labels, score_tensor)
        support = finite_labels & torch.isfinite(score_tensor)
        if not torch.equal(support, clean_support):
            delta, delta_status = float("nan"), "undefined_support_mismatch"
        elif not (math.isfinite(metric["macro_auroc"]) and math.isfinite(clean_metric["macro_auroc"])):
            delta, delta_status = float("nan"), "undefined_auroc"
        else:
            delta = delta_auroc(metric["macro_auroc"], clean_metric["macro_auroc"])
            delta_status = "ok"
        values = torch.cat(drifts)
        finite = torch.isfinite(values)
        rows.append({"model": model_name, "protocol": protocol,
                     "perturbation": name, "severity": severity,
                     "auroc": metric["macro_auroc"],
                     "delta_auroc": delta, "delta_status": delta_status,
                     "drift": values[finite].mean().item() if finite.any() else float("nan"),
                     "valid_classes": metric["valid_classes"], "total_classes": metric["total_classes"],
                     "valid_drift_pairs": int(finite.sum()), "total_pairs": labels.shape[0],
                     "per_class_auroc": metric["per_class_auroc"],
                     "valid_samples_per_class": metric["valid_samples"],
                     "parameters": dict(parameters), "evidence": evidence})
    return rows
