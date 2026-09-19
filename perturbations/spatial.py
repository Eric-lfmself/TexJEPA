"""TexJEPA Sec. IV-B, Fig. 4: noise, blur, brightness, contrast.

Gaussian sigma is in raw [0,1] image units. Clipping noisy images, additive
brightness, spatial per-channel contrast centering, and blur kernel support
are implementation choices because the manuscript does not specify them.
"""

import math

import torch
import torch.nn.functional as F

from ._common import batch_image, restore_image


def gaussian_noise(x, sigma, *, generator=None, clip=True):
    images, single = batch_image(x)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    noise = torch.randn(images.shape, dtype=images.dtype, device=images.device,
                        generator=generator)
    output = images + float(sigma) * noise
    return restore_image(output.clamp(0, 1) if clip else output, single)


def gaussian_blur(x, sigma, *, kernel_size=None, generator=None):
    images, single = batch_image(x)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if sigma == 0:
        return x.clone()
    size = int(2 * math.ceil(3 * sigma) + 1) if kernel_size is None else kernel_size
    if not isinstance(size, int) or size < 1 or size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    coordinates = torch.arange(size, device=images.device, dtype=images.dtype) - size // 2
    weights = torch.exp(-0.5 * (coordinates / sigma).square())
    weights = weights / weights.sum()
    kernel = torch.outer(weights, weights).expand(images.shape[1], 1, size, size)
    padded = F.pad(images, (size // 2,) * 4, mode="replicate")
    result = F.conv2d(padded, kernel, groups=images.shape[1])
    return restore_image(result.clamp(0, 1), single)


def brightness_shift(x, shift, *, generator=None):
    images, single = batch_image(x)
    if not math.isfinite(shift):
        raise ValueError("shift must be finite")
    return restore_image((images + shift).clamp(0, 1), single)


def contrast_scale(x, factor, *, generator=None):
    images, single = batch_image(x)
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("factor must be finite and nonnegative")
    mean = images.mean(dim=(-2, -1), keepdim=True)
    return restore_image((mean + factor * (images - mean)).clamp(0, 1), single)
