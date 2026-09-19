"""Image perturbations for TexJEPA Sec. IV-B/C and Fig. 4.

All functions accept CHW or BCHW floating images in [0, 1]. Noise, blur,
brightness, and contrast produce values in [0, 1] by default. FFT filtering
deliberately returns the un-clipped inverse transform: clipping changes the
frequency energy and would invalidate the intended frequency intervention.
"""

from .spatial import gaussian_noise, gaussian_blur, brightness_shift, contrast_scale
from .frequency import frequency_filter, low_pass, band_pass, band_stop

__all__ = ["gaussian_noise", "gaussian_blur", "brightness_shift", "contrast_scale",
           "frequency_filter", "low_pass", "band_pass", "band_stop"]
