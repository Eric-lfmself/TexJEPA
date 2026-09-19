"""Shared probing training loop (Sec. IV-A, Table III).

Linear: BN then linear, frozen encoder, SGD+cosine. MLP: one 512-dimensional
hidden GELU/dropout layer, frozen encoder. Partial FT: only last two blocks,
final norm and MLP head; AdamW with a lower encoder learning rate. LRs/dropout
not specified in the paper are configurable implementation defaults.
"""
from collections.abc import Iterator
from dataclasses import asdict, dataclass
import copy
from pathlib import Path
import torch
from torch import nn


@dataclass
class ProbeConfig:
    protocol: str = "linear"
    num_classes: int = 15
    hidden_dim: int = 512
    dropout: float = 0.1
    epochs: int | None = None
    head_lr: float = 0.01
    encoder_lr: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.0001

    def __post_init__(self):
        if self.protocol not in ("linear", "mlp", "partial_ft"):
            raise ValueError("Unknown probing protocol")
        if self.epochs is None:
            self.epochs = 15 if self.protocol == "partial_ft" else 100
        if self.epochs < 1 or self.head_lr <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid epochs, learning rate or dropout")
        if self.protocol == "partial_ft" and not 0 < self.encoder_lr < self.head_lr:
            raise ValueError("Partial FT encoder learning rate must be lower than head LR")


class ProbeModel(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder, self.config = encoder, config
        self.encoder.requires_grad_(False)
        if config.protocol == "linear":
            self.head = nn.Sequential(nn.BatchNorm1d(encoder.embed_dim), nn.Linear(encoder.embed_dim, config.num_classes))
        else:
            self.head = nn.Sequential(nn.Linear(encoder.embed_dim, config.hidden_dim), nn.GELU(),
                                      nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.num_classes))
        if config.protocol == "partial_ft":
            if not hasattr(encoder, "blocks") or not hasattr(encoder, "norm") or len(encoder.blocks) < 2:
                raise ValueError("Partial FT requires at least two encoder blocks and final norm")
            for block in encoder.blocks[-2:]:
                block.requires_grad_(True)
            encoder.norm.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode and self.config.protocol == "partial_ft":
            for block in self.encoder.blocks[-2:]:
                block.train(True)
            self.encoder.norm.train(True)
        return self

    def forward(self, images):
        if self.config.protocol == "partial_ft":
            features = self.encoder(images)
        else:
            with torch.no_grad():
                features = self.encoder(images)
        return self.head(features)


def positive_class_weights(labels):
    """Use training labels only; undefined prevalence gets neutral weight 1.

    Returns the weights and affected class indices for transparent reporting.
    Classes with no positives OR no negatives cannot define a ratio useful for
    both-class discrimination and are reported, never silently divided by zero.
    """
    y = torch.as_tensor(labels, dtype=torch.float32, device="cpu")
    if y.ndim != 2 or not len(y) or not torch.all((y == 0) | (y == 1)):
        raise ValueError("Expected nonempty, finite binary training labels")
    positives, negatives = y.sum(0), y.shape[0] - y.sum(0)
    defined = (positives > 0) & (negatives > 0)
    weights = torch.ones_like(positives)
    weights[defined] = negatives[defined] / positives[defined]
    return weights, (~defined).nonzero().flatten().tolist()


def build_optimizer(model):
    c = model.config
    if c.protocol == "partial_ft":
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": c.encoder_lr, "name": "encoder"},
            {"params": model.head.parameters(), "lr": c.head_lr, "name": "head"}], weight_decay=c.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.head.parameters(), lr=c.head_lr,
                                    momentum=c.momentum, weight_decay=c.weight_decay)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)


def fit_probe(encoder, train_batches, config=None, *, validation_batches=None,
              device="cpu", training_labels=None, augmentation=None,
              augmentation_seed=42, checkpoint_dir=None, resume_from=None,
              metadata=None, generators=None, seed=None, checkpoint_every=1,
              stop_after_epoch=None):
    """Shared Table III trainer with explicit device, augmentation and resumption.

    ``augmentation(images, *, generator)`` is train-only and receives raw [0,1]
    images before the encoder's normalizer. Supply complete ``training_labels``
    or a dataset.labels_tensor() method for drop_last/replacement samplers.
    Checkpoints store full encoder/head/BN, optimizer/scheduler and epoch RNG.
    Resumption retains the original planned config.epochs and is supported only
    at complete epoch boundaries; stop_after_epoch can interrupt that schedule.
    Default behavior remains a CPU loop returning (model.eval(), info).
    """
    from training.loops import collect_training_labels, save_epoch
    from training.state import (collect_generators, explicit_device, load_training_state,
                                seed_runtime)

    config = config or ProbeConfig()
    device = explicit_device(device)
    if isinstance(train_batches, Iterator):
        raise ValueError("train_batches must be re-iterable, not a consumed generator")
    if validation_batches is not None and config.epochs > 1 and isinstance(validation_batches, Iterator):
        raise ValueError("Multiple validation epochs require re-iterable batches")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    end_epoch = config.epochs if stop_after_epoch is None else stop_after_epoch
    if type(end_epoch) is not int or not 1 <= end_epoch <= config.epochs:
        raise ValueError("stop_after_epoch must be within the planned epoch schedule")
    if seed is not None:
        seed_runtime(seed, device)
    labels_snapshot, label_source = collect_training_labels(train_batches, training_labels)
    weights, undefined = positive_class_weights(labels_snapshot)
    if weights.numel() != config.num_classes:
        raise ValueError("Training labels do not match configured classes")
    model = ProbeModel(encoder, config).to(device=device)
    optimizer, scheduler = build_optimizer(model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device=device))
    augmentation_generator = torch.Generator(device=device).manual_seed(augmentation_seed)
    rng = collect_generators(train_batches, generators,
                              augmentation=augmentation_generator if augmentation is not None else None)
    if validation_batches is not None:
        for name, generator in collect_generators(validation_batches).items():
            name = "validation_" + name
            if name in rng and rng[name] is not generator:
                raise ValueError(f"Conflicting validation generator: {name}")
            rng[name] = generator
    run_metadata = copy.deepcopy(metadata or {})
    for key in ("epoch", "epoch_is_lineage_label", "actual_training_steps", "global_step"):
        run_metadata.pop(key, None)
    augmentation_name = (f"{getattr(augmentation, '__module__', '')}.{getattr(augmentation, '__qualname__', type(augmentation).__name__)}"
                         if augmentation is not None else None)
    run_metadata["training_spec"] = {"kind": "probe", "config": asdict(config),
                                     "device": str(device), "seed": seed,
                                     "augmentation": augmentation_name,
                                     "augmentation_seed": augmentation_seed,
                                     "pos_weight": weights.tolist(), "weight_samples": len(labels_snapshot)}
    history, start_epoch, global_step = [], 0, 0
    if resume_from is not None:
        state = load_training_state(resume_from, model, optimizer, scheduler, generators=rng,
                                     expected_metadata=run_metadata, device=device)
        history, start_epoch, global_step = state["history"], state["next_epoch"], state["global_step"]
    if start_epoch > end_epoch:
        raise ValueError("Resume epoch exceeds requested training boundary")
    latest = str(Path(resume_from).resolve()) if resume_from else None
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total, count, optimizer_steps = 0., 0, 0
        rates = [group["lr"] for group in optimizer.param_groups]
        for batch in train_batches:
            images = batch["image"].to(device=device)
            labels = batch["labels"].to(device=device, dtype=torch.float32)
            if config.protocol == "linear" and images.shape[0] < 2:
                raise ValueError("BatchNorm linear probe needs >=2 training examples per batch")
            if augmentation is not None:
                images = augmentation(images, generator=augmentation_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probing loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
            optimizer_steps += 1
            global_step += 1
        if not count:
            raise ValueError("Training epoch received no batches")
        row = {"epoch": epoch + 1, "train_loss": total / count,
               "learning_rates": rates, "optimizer_steps": optimizer_steps,
               "global_step": global_step, "samples": count}
        if validation_batches is not None:
            model.eval()
            validation_loss, validation_count = 0., 0
            with torch.no_grad():
                for batch in validation_batches:
                    labels = batch["labels"].to(device=device, dtype=torch.float32)
                    validation_loss += criterion(model(batch["image"].to(device=device)), labels).item() * len(labels)
                    validation_count += len(labels)
            if not validation_count:
                raise ValueError("Validation batches are empty")
            row["validation_loss"] = validation_loss / validation_count
        history.append(row)
        scheduler.step()
        latest = save_epoch(checkpoint_dir, model, optimizer, scheduler, epoch + 1, history,
                             run_metadata, rng, global_step, device=device,
                             checkpoint_every=checkpoint_every, final_epoch=end_epoch) or latest
    return model.eval(), {"protocol": config.protocol, "history": history,
                          "pos_weight": weights.tolist(), "undefined_prevalence_classes": undefined,
                          "weight_source": "train_split_only", "class_weight_label_source": label_source,
                          "class_weight_sample_count": len(labels_snapshot),
                          "actual_training_steps": global_step, "completed_epochs": len(history),
                          "checkpoint": latest, "metadata": run_metadata}
