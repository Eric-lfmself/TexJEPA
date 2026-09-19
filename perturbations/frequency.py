"""FFT interventions for TexJEPA Sec. IV-C (Table VI).

The manuscript states cutoff 0.15 and bands [0,.2], [.2,.45], [.45,1],
but does NOT define radial normalization or endpoint handling. Our explicit
default assumption divides radius (cycles/pixel) by sqrt(.5**2+.5**2),
the 2-D Nyquist corner. ``axis_nyquist`` divides by .5 instead, so corner
frequencies exceed 1. Bands are lower-inclusive/upper-exclusive, except that
the top band includes 1. This is an implementation assumption, not recovered
historical code. Inverse transforms are never silently clipped or rescaled.
"""

import math

import torch

from ._common import batch_image, restore_image


def frequency_filter(x, mode, *, low=0.0, high=1.0,
                     normalization="corner_nyquist", generator=None):
    images, single = batch_image(x)
    if mode not in {"low_pass", "band_pass", "band_stop"}:
        raise ValueError("mode must be low_pass, band_pass, or band_stop")
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 1):
        raise ValueError("Require finite 0 <= low < high <= 1")
    if normalization not in {"corner_nyquist", "axis_nyquist"}:
        raise ValueError("Unknown frequency normalization")
    # FFT on CPU does not support float16; the output dtype still matches input.
    work = images if images.dtype in (torch.float32, torch.float64) else images.float()
    fy = torch.fft.fftfreq(images.shape[-2], device=images.device, dtype=work.dtype)
    fx = torch.fft.fftfreq(images.shape[-1], device=images.device, dtype=work.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radius = radius / (math.sqrt(0.5) if normalization == "corner_nyquist" else 0.5)
    if mode == "low_pass":
        keep = radius <= high
    else:
        upper = radius <= high if high == 1.0 else radius < high
        band = (radius >= low) & upper
        keep = ~band if mode == "band_stop" else band
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    result = torch.fft.ifft2(spectrum * keep, dim=(-2, -1)).real.to(images.dtype)
    return restore_image(result, single)


def low_pass(x, cutoff=0.15, *, normalization="corner_nyquist", generator=None):
    return frequency_filter(x, "low_pass", high=cutoff, normalization=normalization)


def band_pass(x, low, high, *, normalization="corner_nyquist", generator=None):
    return frequency_filter(x, "band_pass", low=low, high=high, normalization=normalization)


def band_stop(x, low, high, *, normalization="corner_nyquist", generator=None):
    return frequency_filter(x, "band_stop", low=low, high=high, normalization=normalization)
