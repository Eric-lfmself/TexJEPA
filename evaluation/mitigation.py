"""Input median/Gaussian smoothing, TexJEPA Sec. V-I, Fig. 8/Table XIII.

Kernel sizes and sigma are configurable implementation choices (not specified
numerically in the paper). Filters preserve constant images and do not clamp
or renormalize; signed FFT-corruption outputs remain meaningful inputs. Clean
and perturbed paths receive identical preprocessing for paired comparison.
"""

import math

import torch
import torch.nn.functional as F

from .robustness import _evaluate_preprocessors


def _batch(x):
    if x.ndim not in (3, 4) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Smoothing expects finite CHW or BCHW floating images")
    return (x.unsqueeze(0), True) if x.ndim == 3 else (x, False)


def _kernel_size(size):
    if not isinstance(size, int) or size < 1 or size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")


def median_filter(x, kernel_size=3):
    images, single = _batch(x)
    _kernel_size(kernel_size)
    if kernel_size == 1:
        return x.clone()
    pad = kernel_size // 2
    padded = F.pad(images, (pad,) * 4, mode="replicate")
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    result = patches.flatten(-2).median(dim=-1).values
    return result.squeeze(0) if single else result


def gaussian_smoothing(x, sigma=1.0, kernel_size=None):
    images, single = _batch(x)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if sigma == 0:
        return x.clone()
    size = 2 * math.ceil(3 * sigma) + 1 if kernel_size is None else kernel_size
    _kernel_size(size)
    coordinates = torch.arange(size, device=images.device, dtype=images.dtype) - size // 2
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    # The outer-product Gaussian is separable; two 1-D convolutions preserve
    # replicate boundaries while avoiding the square kernel's extra work.
    channels, pad = images.shape[1], size // 2
    horizontal = kernel.reshape(1, 1, 1, size).expand(channels, 1, 1, size)
    vertical = kernel.reshape(1, 1, size, 1).expand(channels, 1, size, 1)
    # Filter deviations from one pixel per channel, then restore that constant
    # component. This avoids accumulating a tiny DC bias over two float32
    # passes, preserving constant backgrounds and small impulse energies.
    baseline = images[:, :, :1, :1]
    result = F.conv2d(F.pad(images - baseline, (pad, pad, 0, 0), mode="replicate"),
                      horizontal, groups=channels)
    result = F.conv2d(F.pad(result, (0, 0, pad, pad), mode="replicate"),
                      vertical, groups=channels)
    result = result + baseline
    return result.squeeze(0) if single else result


def evaluate_mitigations(encoder, head, batches, perturbations, *, model_name,
                         protocol="linear", seed=42, evidence="synthetic_smoke",
                         median_kernel=3, gaussian_sigma=1.0,
                         include_unmodified=True):
    """Return paired robustness rows for original/median/Gaussian inputs.

    All methods share one traversal and exactly the same clean/corrupted
    images, so shuffled loaders and one-shot streams remain paired. Set
    include_unmodified=False to evaluate only median/Gaussian when the caller
    already has original rows from the same images, order, batching and seed.
    """
    methods = [("median", lambda x: median_filter(x, median_kernel)),
               ("gaussian", lambda x: gaussian_smoothing(x, gaussian_sigma))]
    if include_unmodified:
        methods.insert(0, ("none", None))
    grouped = _evaluate_preprocessors(
        encoder, head, batches, perturbations, [transform for _, transform in methods],
        model_name=model_name, protocol=protocol, seed=seed, evidence=evidence)
    rows = []
    for (name, _), records in zip(methods, grouped):
        for record in records:
            record["mitigation"] = name
            record["mitigation_parameters"] = ({"kernel_size": median_kernel} if name == "median"
                                                  else {"sigma": gaussian_sigma} if name == "gaussian"
                                                  else {})
        rows.extend(records)
    return rows
