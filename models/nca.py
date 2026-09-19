"""Residual Noise-Consistency Adapter, TexJEPA Sec. V-I, Fig. 8/Table XIII.

The manuscript describes a small residual adapter after a frozen encoder,
aligning clean/augmented representations and predictions, but does not specify
layers, loss coefficients, or optimizer settings. This explicit configurable
skeleton uses a GELU bottleneck residual MLP, representation MSE, probability
MSE, and optional supervised BCE. These are implementation assumptions, not
recovered historical hyperparameters. Consistency alone can collapse: the
default supervised anchor is enabled and synthetic scores prove no accuracy.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from perturbations import gaussian_noise


class ResidualAdapter(nn.Module):
    """Identity-initialized z + W2 GELU(W1 LayerNorm(z))."""

    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.network = nn.Sequential(nn.LayerNorm(feature_dim),
                                     nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, feature_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return features + self.network(features)


class NoiseConsistencyAdapter(nn.Module):
    """Frozen backbone + trainable residual adapter and multilabel head."""

    def __init__(self, encoder, feature_dim, num_classes=15, hidden_dim=64, head=None):
        super().__init__()
        self.encoder = encoder
        self.adapter = ResidualAdapter(feature_dim, hidden_dim)
        self.head = nn.Linear(feature_dim, num_classes) if head is None else head
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # Preserve backbone dropout/BatchNorm state as well as its parameters.
        self.encoder.eval()
        return self

    def encode(self, images):
        with torch.no_grad():
            features = self.encoder(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("NCA encoder must return pooled BxD features")
        return self.adapter(features.detach())

    def forward(self, images):
        return self.head(self.encode(images))


def nca_objective(clean_features, augmented_features, clean_logits,
                  augmented_logits, labels=None, *, alignment_weight=1.0,
                  consistency_weight=1.0, supervised_weight=1.0,
                  pos_weight=None):
    """Configurable clean/augmented alignment, consistency, supervised anchor."""
    weights = (alignment_weight, consistency_weight, supervised_weight)
    if not all(math.isfinite(value) and value >= 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("Loss weights must be finite, nonnegative, and not all zero")
    if clean_features.ndim != 2 or clean_features.shape != augmented_features.shape:
        raise ValueError("Clean and augmented representations must be matching BxD tensors")
    if clean_logits.ndim != 2 or clean_logits.shape != augmented_logits.shape:
        raise ValueError("Clean and augmented logits must be matching BxC tensors")
    alignment = F.mse_loss(clean_features, augmented_features)
    consistency = F.mse_loss(clean_logits.sigmoid(), augmented_logits.sigmoid())
    supervised = clean_logits.sum() * 0
    if supervised_weight > 0:
        if labels is None or labels.shape != clean_logits.shape:
            raise ValueError("Supervised anchor requires labels matching logits")
        labels = labels.to(device=clean_logits.device, dtype=clean_logits.dtype)
        if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("NCA supervision requires finite binary labels")
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, device=clean_logits.device, dtype=clean_logits.dtype)
        supervised = (F.binary_cross_entropy_with_logits(clean_logits, labels, pos_weight=pos_weight)
                      + F.binary_cross_entropy_with_logits(augmented_logits, labels, pos_weight=pos_weight)) / 2
    loss = alignment_weight * alignment + consistency_weight * consistency + supervised_weight * supervised
    return {"loss": loss, "alignment": alignment,
            "consistency": consistency, "supervised": supervised}


def train_nca(model, batches, *, epochs=1, learning_rate=1e-3, noise_sigma=0.05,
              alignment_weight=1.0, consistency_weight=1.0, supervised_weight=1.0,
              pos_weight=None, seed=42, max_steps=None):
    """Shared tiny/real-loader skeleton; trains only adapter/head parameters.

    Batch dicts contain image [B,C,H,W] and labels [B,K]. Device placement is
    explicitly caller-controlled (smoke callers use CPU); no GPU detection,
    downloads, or pretrained-weight resolution occur. For epochs > 1, pass a
    re-iterable DataLoader/list. max_steps is a total-step bound across epochs.
    """
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if epochs > 1 and iter(batches) is batches:
        raise ValueError("Multiple epochs require a re-iterable batch collection")
    model.train()
    model.encoder.requires_grad_(False)
    parameters = [*model.adapter.parameters(), *model.head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    generator, history = None, []
    for epoch in range(epochs):
        epoch_steps = 0
        for batch in batches:
            images = batch["image"]
            if generator is None:
                generator = torch.Generator(device=images.device).manual_seed(seed)
            augmented = gaussian_noise(images, noise_sigma, generator=generator)
            clean_features = model.encode(images)
            augmented_features = model.encode(augmented)
            losses = nca_objective(clean_features, augmented_features,
                                    model.head(clean_features), model.head(augmented_features),
                                    batch.get("labels"), alignment_weight=alignment_weight,
                                    consistency_weight=consistency_weight,
                                    supervised_weight=supervised_weight, pos_weight=pos_weight)
            if not torch.isfinite(losses["loss"]):
                raise ValueError("Nonfinite NCA objective; refusing an invalid update")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            history.append({"epoch": epoch, "step": len(history),
                            **{key: value.detach().item() for key, value in losses.items()}})
            epoch_steps += 1
            if max_steps is not None and len(history) >= max_steps:
                return history
        if epoch_steps == 0:
            raise ValueError("NCA epoch received no batches")
    return history
