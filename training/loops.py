"""Complete local post-training and NCA loops, Sec. IV-E / V-I.

Settings absent from the manuscript remain explicit implementation defaults.
Only caller-supplied models/data are used. Epoch checkpoints can be resumed or
loaded for evaluation; no pretrained weights or datasets are resolved online.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path

import torch
from torch.utils.data import Subset

from models.nca import nca_objective
from perturbations import gaussian_noise
from .state import (collect_generators, explicit_device, load_training_state,
                    save_training_state, seed_runtime)


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    checkpoint_every: int = 1
    scheduler: str = "cosine"
    ema_start: float = .996
    ema_end: float = 1.0
    max_steps: int | None = None

    def __post_init__(self):
        for name in ("epochs", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        for name in ("learning_rate", "weight_decay", "ema_start", "ema_end"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay nonnegative")
        if not 0 <= self.ema_start <= self.ema_end <= 1:
            raise ValueError("EMA schedule must satisfy 0 <= start <= end <= 1")
        if self.scheduler not in ("cosine", "none"):
            raise ValueError("scheduler must be cosine or none")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        self.device = str(explicit_device(self.device))


def _dataset_labels(dataset):
    if isinstance(dataset, Subset):
        labels = _dataset_labels(dataset.dataset)
        return None if labels is None else labels[dataset.indices]
    if callable(getattr(dataset, "labels_tensor", None)):
        return torch.as_tensor(dataset.labels_tensor()).detach().cpu().clone()
    return None


def collect_training_labels(batches, training_labels=None):
    """Compute class weights from the complete training split, not sampled epochs.

    Explicit labels or Dataset.labels_tensor() avoid image I/O and drop_last /
    replacement-sampler prevalence bias. Plain batch collections remain supported.
    """
    if training_labels is not None:
        return torch.as_tensor(training_labels).detach().cpu().clone(), "explicit_training_labels"
    labels = _dataset_labels(getattr(batches, "dataset", None))
    if labels is not None:
        return labels, "dataset_metadata"
    if getattr(batches, "drop_last", False):
        raise ValueError("drop_last loader needs complete training_labels or dataset.labels_tensor()")
    sampler = getattr(batches, "sampler", None)
    if sampler is not None:
        from torch.utils.data import RandomSampler, SequentialSampler
        if not isinstance(sampler, (RandomSampler, SequentialSampler)) or getattr(sampler, "replacement", False):
            raise ValueError("Custom/replacement sampler needs complete training_labels metadata")
        if isinstance(sampler, RandomSampler) and sampler.num_samples != len(batches.dataset):
            raise ValueError("Subsampled loader needs complete training_labels metadata")
    if isinstance(batches, Iterator):
        raise ValueError("Training batches must be re-iterable")
    parts = [batch["labels"].detach().cpu().clone() for batch in batches]
    if not parts:
        raise ValueError("No training labels")
    return torch.cat(parts), "observed_training_batches"


def _epoch_count(batches, config, stop_after_epoch):
    if isinstance(batches, Iterator):
        raise ValueError("Epoch training needs re-iterable batches")
    try:
        steps = len(batches)
    except TypeError as exc:
        raise ValueError("Epoch training needs a sized DataLoader or batch collection") from exc
    if steps < 1:
        raise ValueError("No training batches")
    end = config.epochs
    if config.max_steps is not None:
        if config.max_steps % steps:
            raise ValueError("max_steps must end at a full epoch boundary")
        end = min(end, config.max_steps // steps)
    if stop_after_epoch is not None:
        if type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= config.epochs:
            raise ValueError("stop_after_epoch must be within the planned epoch schedule")
        end = min(end, stop_after_epoch)
    return steps, end


def _scheduler(optimizer, config):
    return (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
            if config.scheduler == "cosine" else None)


def _metadata(metadata, config, kind, **details):
    result = copy.deepcopy(metadata or {})
    for key in ("epoch", "epoch_is_lineage_label", "actual_training_steps", "global_step"):
        result.pop(key, None)
    setting = asdict(config)
    # These affect stopping/storage, not the mathematical training trajectory.
    setting.pop("max_steps")
    setting.pop("checkpoint_every")
    result["training_spec"] = {"kind": kind, "config": setting, **details}
    return result


def _resume(path, model, optimizer, scheduler, generators, metadata, device):
    if path is None:
        return 0, 0, []
    state = load_training_state(path, model, optimizer, scheduler, generators=generators,
                                expected_metadata=metadata, device=device)
    return state["next_epoch"], state["global_step"], state["history"]


def save_epoch(checkpoint_dir, model, optimizer, scheduler, epoch, history, metadata,
               generators, global_step, *, device, checkpoint_every=1, final_epoch=None):
    """Publish native-compatible epoch_NNNN.pt and latest.pt checkpoints.

    The true v4 epoch-50 node is retained regardless of the regular save interval.
    No epoch label is synthesized from a shorter run.
    """
    if checkpoint_dir is None:
        return None
    folder = Path(checkpoint_dir)
    if epoch % checkpoint_every == 0 or epoch == final_epoch or epoch == 50:
        save_training_state(folder / f"epoch_{epoch:04d}.pt", model, optimizer, scheduler,
                            epoch, history, metadata, generators, global_step, device=device)
    latest = save_training_state(folder / "latest.pt", model, optimizer, scheduler,
                                 epoch, history, metadata, generators, global_step, device=device)
    return str(latest.resolve())


def _objective_spec(objective):
    if objective is None:
        return {"mode": "baseline"}
    fields = ("mode", "noise_types", "gaussian_sigma", "poisson_peak", "jpeg_quality",
              "lambda_var", "lambda_cov", "variance_gamma", "variance_eps")
    return {key: list(value) if isinstance(value, tuple) else value
            for key in fields if (value := getattr(objective, key, None)) is not None}


def train_post_training(model, objective, batches, config=None, *, mask_sampler,
                        checkpoint_dir=None, resume_from=None, metadata=None,
                        generators=None, stop_after_epoch=None):
    """Train I-JEPA/v4/v5/v6 for actual epochs; EMA follows successful updates.

    ``objective=None`` supports the baseline I-JEPA latent loss. Mask geometry
    uses an independent CPU generator; corruption uses the explicit image device.
    ``stop_after_epoch`` permits a resumable interruption of the original planned
    schedule. Return ``(model.eval(), info)``; all checkpoint epochs are real.
    """
    config = config or TrainingConfig()
    steps, end_epoch = _epoch_count(batches, config, stop_after_epoch)
    device = explicit_device(config.device)
    seed_runtime(config.seed, device)
    model.to(device)
    if objective is not None:
        objective.to(device)
    noise = torch.Generator(device=device).manual_seed(config.seed + 1)
    rng = collect_generators(batches, generators, noise=noise, mask=mask_sampler.generator)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = _scheduler(optimizer, config)
    mask_keys = ("grid_size", "context_scale", "target_scale", "num_target_blocks",
                 "target_aspect_ratio", "max_attempts")
    mask_spec = {key: list(value) if isinstance(value, tuple) else value
                 for key in mask_keys if (value := getattr(mask_sampler, key, None)) is not None}
    run_metadata = _metadata(metadata, config, "post_training", steps_per_epoch=steps,
                             objective=_objective_spec(objective), mask=mask_spec)
    start, global_step, history = _resume(resume_from, model, optimizer, scheduler, rng, run_metadata, device)
    if start > config.epochs or start > end_epoch:
        raise ValueError("Resume epoch exceeds the requested training boundary")
    latest = str(Path(resume_from).resolve()) if resume_from else None
    total_steps = config.epochs * steps
    for epoch in range(start, end_epoch):
        model.train()
        count, observed_steps, totals, noises, fallback_batches = 0, 0, {}, {}, 0
        rates = [group["lr"] for group in optimizer.param_groups]
        for batch in batches:
            images = batch["image"].to(device=device)
            masks = mask_sampler(len(images), device=device)
            fallback_batches += int(masks["metadata"].get("fallback_used", False))
            if objective is None:
                losses = model(images, masks["context_indices"], masks["target_indices"])
            else:
                losses = objective(model, images, masks["context_indices"], masks["target_indices"], generator=noise)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite post-training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            momentum = config.ema_start + (config.ema_end - config.ema_start) * ((global_step - 1) / max(1, total_steps - 1))
            model.update_ema(momentum)
            for key in ("loss", "asymmetric_loss", "variance_loss", "covariance_loss"):
                if key in losses:
                    totals[key] = totals.get(key, 0.) + float(losses[key].detach()) * len(images)
            noise_kind = losses.get("noise_kind", "none")
            noises[noise_kind] = noises.get(noise_kind, 0) + 1
            count += len(images)
            observed_steps += 1
        if observed_steps != steps or not count:
            raise ValueError("Batch count changed within a planned epoch; no epoch checkpoint published")
        history.append({"epoch": epoch + 1, "optimizer_steps": observed_steps, "global_step": global_step,
                        "samples": count, "learning_rates": rates, "ema_momentum": momentum,
                        "noise_batches": noises, "mask_fallback_batches": fallback_batches,
                        **{key: value / count for key, value in totals.items()}})
        if scheduler is not None:
            scheduler.step()
        latest = save_epoch(checkpoint_dir, model, optimizer, scheduler, epoch + 1, history,
                             run_metadata, rng, global_step, device=device,
                             checkpoint_every=config.checkpoint_every, final_epoch=end_epoch) or latest
    return model.eval(), {"history": history, "completed_epochs": len(history),
                           "actual_training_steps": global_step, "checkpoint": latest,
                           "planned_epochs": config.epochs, "metadata": run_metadata}


def train_nca_resumable(model, batches, config=None, *, noise_sigma=.05,
                        alignment_weight=1., consistency_weight=1., supervised_weight=1.,
                        pos_weight=None, training_labels=None, checkpoint_dir=None,
                        resume_from=None, metadata=None, generators=None, stop_after_epoch=None):
    """Train only NCA adapter/head with complete epoch, seed and resume support."""
    config = config or TrainingConfig()
    steps, end_epoch = _epoch_count(batches, config, stop_after_epoch)
    device = explicit_device(config.device)
    seed_runtime(config.seed, device)
    model.to(device)
    model.encoder.requires_grad_(False)
    if not math.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise_sigma must be finite and nonnegative")
    loss_weights = (alignment_weight, consistency_weight, supervised_weight)
    if any(not math.isfinite(v) or v < 0 for v in loss_weights) or not any(loss_weights):
        raise ValueError("NCA loss weights must be finite, nonnegative and not all zero")
    weight_source, undefined = "caller_supplied" if pos_weight is not None else None, []
    if supervised_weight > 0 and pos_weight is None:
        from probing.trainer import positive_class_weights
        labels, weight_source = collect_training_labels(batches, training_labels)
        pos_weight, undefined = positive_class_weights(labels)
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
        if pos_weight.ndim != 1 or not torch.isfinite(pos_weight).all() or (pos_weight < 0).any():
            raise ValueError("NCA pos_weight must be a finite nonnegative class vector")
    noise = torch.Generator(device=device).manual_seed(config.seed + 1)
    rng = collect_generators(batches, generators, noise=noise)
    optimizer = torch.optim.AdamW([*model.adapter.parameters(), *model.head.parameters()],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = _scheduler(optimizer, config)
    run_metadata = _metadata(metadata, config, "nca", steps_per_epoch=steps, noise_sigma=noise_sigma,
                             alignment_weight=alignment_weight, consistency_weight=consistency_weight,
                             supervised_weight=supervised_weight,
                             pos_weight=None if pos_weight is None else pos_weight.cpu().tolist())
    start, global_step, history = _resume(resume_from, model, optimizer, scheduler, rng, run_metadata, device)
    if start > config.epochs or start > end_epoch:
        raise ValueError("Resume epoch exceeds the requested training boundary")
    latest = str(Path(resume_from).resolve()) if resume_from else None
    for epoch in range(start, end_epoch):
        model.train()
        count, observed_steps, totals = 0, 0, {}
        rates = [group["lr"] for group in optimizer.param_groups]
        for batch in batches:
            images = batch["image"].to(device=device)
            labels = batch.get("labels")
            labels = None if labels is None else labels.to(device=device)
            augmented = gaussian_noise(images, noise_sigma, generator=noise)
            clean_features, augmented_features = model.encode(images), model.encode(augmented)
            loss_values = nca_objective(clean_features, augmented_features,
                                       model.head(clean_features), model.head(augmented_features), labels,
                                       alignment_weight=alignment_weight, consistency_weight=consistency_weight,
                                       supervised_weight=supervised_weight, pos_weight=pos_weight)
            if not torch.isfinite(loss_values["loss"]):
                raise FloatingPointError("Nonfinite NCA objective")
            optimizer.zero_grad(set_to_none=True)
            loss_values["loss"].backward()
            optimizer.step()
            global_step += 1
            for key, value in loss_values.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * len(images)
            count += len(images)
            observed_steps += 1
        if observed_steps != steps or not count:
            raise ValueError("Batch count changed within a planned epoch; no epoch checkpoint published")
        history.append({"epoch": epoch + 1, "optimizer_steps": observed_steps, "global_step": global_step,
                        "samples": count, "learning_rates": rates,
                        **{key: value / count for key, value in totals.items()}})
        if scheduler is not None:
            scheduler.step()
        latest = save_epoch(checkpoint_dir, model, optimizer, scheduler, epoch + 1, history,
                             run_metadata, rng, global_step, device=device,
                             checkpoint_every=config.checkpoint_every, final_epoch=end_epoch) or latest
    return model.eval(), {"history": history, "completed_epochs": len(history),
                           "actual_training_steps": global_step, "checkpoint": latest,
                           "planned_epochs": config.epochs, "metadata": run_metadata,
                           "pos_weight": None if pos_weight is None else pos_weight.cpu().tolist(),
                           "weight_source": "train_split_only" if supervised_weight else None,
                           "class_weight_label_source": weight_source,
                           "undefined_prevalence_classes": undefined}


def train_pretraining(model, batches, config=None, *, kind="ijepa", mask_sampler=None,
                      mask_ratio=.75, normalizer=None, checkpoint_dir=None, resume_from=None,
                      metadata=None, generators=None, stop_after_epoch=None):
    """Optional native I-JEPA/MAE training; normal execution may use supplied weights.

    No paper-scale model is created here. I-JEPA reuses the shared EMA loop;
    MAE uses its masked-pixel loss. All epochs, optimizer state and masks resume
    under the same native checkpoint contract as downstream post-training.
    """
    config = config or TrainingConfig()
    if kind == "ijepa":
        if mask_sampler is None:
            raise ValueError("I-JEPA pretraining requires an explicit mask_sampler")
        objective = None
        if normalizer is not None:
            class NormalizedBaseline(torch.nn.Module):
                mode = "baseline"
                def __init__(self, normalize):
                    super().__init__()
                    self.normalizer = normalize
                def forward(self, current_model, images, context_indices, target_indices, *, generator=None):
                    return current_model(self.normalizer(images), context_indices, target_indices)
            objective = NormalizedBaseline(normalizer)
        return train_post_training(model, objective, batches, config, mask_sampler=mask_sampler,
                                   checkpoint_dir=checkpoint_dir, resume_from=resume_from,
                                   metadata=metadata, generators=generators, stop_after_epoch=stop_after_epoch)
    if kind != "mae":
        raise ValueError("Pretraining kind must be ijepa or mae")
    if not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
        raise ValueError("MAE mask_ratio must be strictly between zero and one")
    steps, end_epoch = _epoch_count(batches, config, stop_after_epoch)
    device = explicit_device(config.device)
    seed_runtime(config.seed, device)
    model.to(device)
    if isinstance(normalizer, torch.nn.Module):
        normalizer.to(device)
    mask_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    rng = collect_generators(batches, generators, mae_mask=mask_generator)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = _scheduler(optimizer, config)
    run_metadata = _metadata(metadata, config, "mae_pretraining", steps_per_epoch=steps,
                             mask_ratio=mask_ratio, norm_pix_loss=bool(model.norm_pix_loss))
    start, global_step, history = _resume(resume_from, model, optimizer, scheduler, rng, run_metadata, device)
    if start > end_epoch:
        raise ValueError("Resume epoch exceeds requested training boundary")
    latest = str(Path(resume_from).resolve()) if resume_from else None
    for epoch in range(start, end_epoch):
        model.train()
        total, count, observed_steps = 0., 0, 0
        rates = [group["lr"] for group in optimizer.param_groups]
        for batch in batches:
            images = batch["image"].to(device=device)
            images = images if normalizer is None else normalizer(images)
            result = model(images, mask_ratio=mask_ratio, generator=mask_generator)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError("Nonfinite MAE reconstruction loss")
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            optimizer.step()
            global_step += 1
            total += float(result["loss"].detach()) * len(images)
            count += len(images)
            observed_steps += 1
        if observed_steps != steps or not count:
            raise ValueError("Batch count changed within a planned epoch; no epoch checkpoint published")
        history.append({"epoch": epoch + 1, "optimizer_steps": observed_steps, "global_step": global_step,
                        "samples": count, "learning_rates": rates, "loss": total / count})
        if scheduler is not None:
            scheduler.step()
        latest = save_epoch(checkpoint_dir, model, optimizer, scheduler, epoch + 1, history,
                             run_metadata, rng, global_step, device=device,
                             checkpoint_every=config.checkpoint_every, final_epoch=end_epoch) or latest
    return model.eval(), {"history": history, "completed_epochs": len(history),
                           "actual_training_steps": global_step, "checkpoint": latest,
                           "planned_epochs": config.epochs, "metadata": run_metadata}
