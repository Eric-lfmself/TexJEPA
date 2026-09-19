"""Explicit image-domain contract for the Sec. IV-B/C interventions."""

import torch


def batch_image(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4):
        raise ValueError("Expected a CHW or BCHW image tensor")
    if not x.is_floating_point() or min(x.shape) < 1:
        raise ValueError("Expected a nonempty floating image tensor")
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Perturbations require finite image intensities in [0, 1]")
    return (x.unsqueeze(0), True) if x.ndim == 3 else (x, False)


def restore_image(x: torch.Tensor, single: bool) -> torch.Tensor:
    return x.squeeze(0) if single else x
