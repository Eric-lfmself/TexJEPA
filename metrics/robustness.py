"""TexJEPA Sec. IV-B: Delta AUROC (Eq. 1), cosine drift (Eq. 2)."""

import torch


def delta_auroc(perturbed, clean):
    """A performance drop is negative, in accordance with manuscript Eq. 1."""
    return perturbed - clean


def cosine_drift(clean, perturbed, *, eps=1e-12):
    """Return drift per row; zero/nonfinite vectors are explicitly NaN.

    A zero vector has undefined angle and cannot certify stability. Inputs
    must be BxD; token sequences must first use an explicitly chosen pooling.
    Overflowed norms are also undefined and excluded from valid drift pairs.
    """
    if clean.ndim != 2 or clean.shape != perturbed.shape:
        raise ValueError("Expected equal BxD clean and perturbed representations")
    if not clean.is_floating_point() or not perturbed.is_floating_point():
        raise ValueError("Representations must be floating tensors")
    if eps <= 0:
        raise ValueError("eps must be positive")
    a, b = clean.to(torch.float64), perturbed.to(torch.float64)
    na, nb = a.norm(dim=1), b.norm(dim=1)
    valid = (torch.isfinite(a).all(dim=1) & torch.isfinite(b).all(dim=1)
             & torch.isfinite(na) & torch.isfinite(nb) & (na > eps) & (nb > eps))
    result = torch.full_like(na, float("nan"))
    result[valid] = 1 - ((a[valid] / na[valid, None]) * (b[valid] / nb[valid, None])).sum(1).clamp(-1, 1)
    return result


def drift_summary(clean, perturbed, *, eps=1e-12):
    values = cosine_drift(clean, perturbed, eps=eps)
    finite = torch.isfinite(values)
    return {"drift": values[finite].mean().item() if finite.any() else float("nan"),
            "valid_pairs": int(finite.sum()), "total_pairs": values.numel()}
