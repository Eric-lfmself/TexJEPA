"""TexJEPA post-training objectives: paper Eqs. (4)--(5), Sec. IV-E.

All variants retain noisy context -> clean, detached EMA targets. v5 additionally
requires four register tokens. v6 adds std-hinge and off-diagonal covariance over
M=batch*visible_patches context samples. It adds no second invariance term.
Noise strengths, JPEG quality, and covariance normalization are explicit
implementation choices where the supplied paper leaves details unspecified.
"""
from __future__ import annotations

import io
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .masking import BlockMaskSampler


def _finite_parameter(name, value, *, positive=False):
    """Reject invalid numeric configuration before tensors/losses are built."""
    try:
        valid = not isinstance(value, bool) and math.isfinite(value)
        valid = valid and (value > 0 if positive else value >= 0)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        domain = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {domain}")


def _jpeg_quality(value):
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("jpeg_quality must be an integer in [1,100]")


def context_noise(images: Tensor, kind="gaussian", *, sigma=0.05, poisson_peak=30.0,
                  jpeg_quality=75, generator=None) -> Tensor:
    """Apply one corruption to [0,1] images; output stays on the input device.

Apply before mean/std normalization. JPEG is a nondifferentiable input transform;
its pixels still feed a differentiable context encoder. No files are written.
"""
    if not images.is_floating_point() or images.ndim != 4:
        raise ValueError("Noise input must be a floating point BCHW image tensor")
    if not bool(torch.isfinite(images).all()) or bool((images < 0).any()) or bool((images > 1).any()):
        raise ValueError("Context noise expects unnormalized image values in [0,1]")
    if kind == "none":
        return images.clone()
    if kind == "gaussian":
        _finite_parameter("Gaussian sigma", sigma)
        return (images + sigma * torch.randn(images.shape, dtype=images.dtype, device=images.device,
                                             generator=generator)).clamp(0, 1)
    if kind == "poisson":
        _finite_parameter("poisson_peak", poisson_peak, positive=True)
        return (torch.poisson(images * poisson_peak, generator=generator) / poisson_peak).clamp(0, 1)
    if kind == "jpeg":
        _jpeg_quality(jpeg_quality)
        if images.shape[1] not in (1, 3):
            raise ValueError("JPEG supports one-channel or three-channel images")
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise ImportError("JPEG corruption requires Pillow and NumPy") from exc
        output = []
        for item in images.detach().cpu():
            array = (item.permute(1, 2, 0).float().numpy() * 255).round().astype("uint8")
            if array.shape[-1] == 1:
                array = array[..., 0]
            image = Image.fromarray(array)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                restored = np.array(decoded, copy=True)
            if restored.ndim == 2:
                restored = restored[..., None]
            output.append(torch.from_numpy(restored).permute(2, 0, 1))
        return torch.stack(output).to(device=images.device, dtype=images.dtype) / 255
    raise ValueError(f"Unknown context noise operator: {kind}")


def patch_variance_covariance(tokens: Tensor, *, gamma=1.0, eps=1e-4) -> dict:
    """VICReg auxiliary terms on M x D (M=batch*patches), Eq. (5).

var = mean(relu(gamma - sqrt(unbiased_var(z)+eps)))
cov = sum(offdiag(cov(z))**2) / D, cov centered with denominator M-1.
    """
    _finite_parameter("variance gamma", gamma, positive=True)
    _finite_parameter("variance eps", eps, positive=True)
    if tokens.ndim not in (2, 3):
        raise ValueError("Patch tokens must have shape (M,D) or (B,N,D)")
    if not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all()):
        raise ValueError("Patch tokens must contain finite floating point values")
    if tokens.shape[-1] < 1:
        raise ValueError("Variance/covariance needs at least two patch samples and one feature")
    z = tokens.reshape(-1, tokens.shape[-1]).float()
    m, d = z.shape
    if m < 2 or d < 1:
        raise ValueError("Variance/covariance needs at least two patch samples and one feature")
    centered = z - z.mean(dim=0)
    std = (centered.square().sum(0) / (m - 1) + eps).sqrt()
    variance = F.relu(gamma - std).mean()
    covariance_matrix = centered.T @ centered / (m - 1)
    off_diagonal = covariance_matrix - torch.diag_embed(torch.diagonal(covariance_matrix))
    covariance = off_diagonal.square().sum() / d
    return {"variance": variance, "covariance": covariance, "std": std,
            "num_samples": m, "feature_dim": d}


class PostTrainingObjective(nn.Module):
    def __init__(self, mode="noise", *, noise_types=("gaussian", "poisson", "jpeg"),
                 gaussian_sigma=0.05, poisson_peak=30.0, jpeg_quality=75,
                 lambda_var=1.0, lambda_cov=0.04, variance_gamma=1.0,
                 variance_eps=1e-4, normalizer=None):
        super().__init__()
        aliases = {"v4": "noise", "v5": "register", "v6": "vicreg"}
        mode = aliases.get(mode, mode)
        if mode not in ("noise", "register", "vicreg"):
            raise ValueError("mode must be noise, register or vicreg")
        if not noise_types or any(kind not in ("gaussian", "poisson", "jpeg", "none") for kind in noise_types):
            raise ValueError("noise_types must contain recognized context noise operators")
        for name, value in (("gaussian_sigma", gaussian_sigma),
                            ("lambda_var", lambda_var), ("lambda_cov", lambda_cov)):
            _finite_parameter(name, value)
        for name, value in (("poisson_peak", poisson_peak),
                            ("variance_gamma", variance_gamma), ("variance_eps", variance_eps)):
            _finite_parameter(name, value, positive=True)
        _jpeg_quality(jpeg_quality)
        self.mode, self.noise_types = mode, tuple(noise_types)
        self.gaussian_sigma, self.poisson_peak, self.jpeg_quality = gaussian_sigma, poisson_peak, jpeg_quality
        self.lambda_var, self.lambda_cov = lambda_var, lambda_cov
        self.variance_gamma, self.variance_eps = variance_gamma, variance_eps
        self.normalizer = normalizer

    def make_mask_sampler(self, grid_size, **kwargs):
        return BlockMaskSampler(grid_size, variant=self.mode, **kwargs)

    def forward(self, model, images: Tensor, context_indices, target_indices, *,
                noise_kind=None, generator=None) -> dict:
        if self.mode == "register" and getattr(model.context_encoder, "num_register_tokens", 0) != 4:
            raise ValueError("TexJEPA-R / v5 requires exactly four register tokens")
        if noise_kind is None:
            # Draw on the same device as the image generator if one is supplied.
            choice_device = generator.device if generator is not None else "cpu"
            choice = int(torch.randint(len(self.noise_types), (), generator=generator, device=choice_device))
            noise_kind = self.noise_types[choice]
        noisy = context_noise(images, noise_kind, sigma=self.gaussian_sigma,
                              poisson_peak=self.poisson_peak, jpeg_quality=self.jpeg_quality,
                              generator=generator)
        clean_input = images if self.normalizer is None else self.normalizer(images)
        context_input = noisy if self.normalizer is None else self.normalizer(noisy)
        result = model(clean_input, context_indices, target_indices, context_images=context_input)
        asymmetric_loss = result["loss"]
        zero = asymmetric_loss.new_zeros(())
        variance, covariance = zero, zero
        regularization = None
        if self.mode == "vicreg":
            regularization = patch_variance_covariance(result["context_tokens"],
                                                       gamma=self.variance_gamma, eps=self.variance_eps)
            variance, covariance = regularization["variance"], regularization["covariance"]
        total = asymmetric_loss + self.lambda_var*variance + self.lambda_cov*covariance
        return {**result, "loss": total, "asymmetric_loss": asymmetric_loss,
                "variance_loss": variance, "covariance_loss": covariance,
                "regularization": regularization, "noise_kind": noise_kind,
                "context_images": noisy, "target_images": images,
                "mode": self.mode}
