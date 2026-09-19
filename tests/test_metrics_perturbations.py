"""Numerical CPU checks of TexJEPA Eqs. 1-2 and frequency interventions."""

import math

import torch
from torch import nn

from evaluation.robustness import PerturbationSpec, evaluate_robustness
from metrics import binary_auroc, multilabel_auroc, cosine_drift, delta_auroc
from perturbations import (band_pass, band_stop, brightness_shift, contrast_scale,
                           gaussian_blur, gaussian_noise, low_pass)


def test_auroc_ties_single_class_and_missing():
    assert binary_auroc([0, 1, 0, 1], [0, 0, 1, 1]) == 0.5
    assert binary_auroc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert binary_auroc([0, 1], [1, 0]) == 0.0
    scores = torch.tensor([[0., 1.], [1., float("nan")]])
    metric = multilabel_auroc(torch.tensor([[0., 1.], [1., 1.]]), scores)
    assert metric["macro_auroc"] == 1.0 and metric["valid_classes"] == 1
    assert math.isnan(metric["per_class_auroc"][1])
    assert metric["valid_samples"] == [2, 1]
    assert math.isnan(multilabel_auroc(torch.ones(2, 1), torch.ones(2, 1))["macro_auroc"])


def test_delta_and_drift_geometry_zero_vector():
    assert abs(delta_auroc(0.641, 0.910) + 0.269) < 1e-12
    a = torch.tensor([[1., 0], [1., 0], [0., 0], [1., 0]])
    b = torch.tensor([[0., 1], [-1., 0], [0., 0], [2., 0]])
    drift = cosine_drift(a, b)
    assert drift[[0, 1, 3]].tolist() == [1.0, 2.0, 0.0]
    assert torch.isnan(drift[2])


def test_spatial_identities_seed_and_numeric_effects():
    x = torch.full((2, 1, 32, 32), 0.5, device="cpu")
    assert torch.equal(gaussian_noise(x, 0), x)
    noisy = gaussian_noise(x, 0.05, generator=torch.Generator().manual_seed(4))
    assert torch.equal(noisy, gaussian_noise(x, 0.05, generator=torch.Generator().manual_seed(4)))
    assert 0.045 < (noisy - x).std().item() < 0.055
    assert torch.allclose(gaussian_blur(x, 1), x, atol=1e-6)
    assert torch.allclose(brightness_shift(x, 0.1), x + 0.1)
    ramp = torch.linspace(0, 1, 32).expand(1, 32, 32)
    assert torch.allclose(contrast_scale(ramp, 0), torch.full_like(ramp, 0.5))
    assert torch.equal(contrast_scale(ramp, 1), ramp)


def test_fft_numeric_band_removal_and_no_clipping():
    # 8 cycles / 32 pixels has corner-normalized radius .35355 (middle band).
    phase = torch.arange(32) * (2 * math.pi * 8 / 32)
    texture = (0.5 + 0.25 * torch.cos(phase)).expand(2, 1, 32, 32).clone()
    assert torch.allclose(low_pass(texture, 0.15), torch.full_like(texture, 0.5), atol=1e-6)
    assert torch.allclose(band_stop(texture, 0.2, 0.45), torch.full_like(texture, 0.5), atol=1e-6)
    isolated = band_pass(texture, 0.2, 0.45)
    assert torch.allclose(isolated, texture - 0.5, atol=1e-6)
    assert isolated.min() < 0  # Negative inverse-FFT values must not be clipped.
    bands = sum(band_pass(texture, lo, hi) for lo, hi in ((0, .2), (.2, .45), (.45, 1)))
    assert torch.allclose(bands, texture, atol=1e-6)


def test_frozen_paired_evaluation_reproducible():
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 4)).cpu()
    head = nn.Linear(4, 2).cpu()
    encoder.train()
    weights = {k: v.clone() for k, v in encoder.state_dict().items()}
    batch = {"image": torch.rand(4, 1, 32, 32), "labels": torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])}
    specs = [PerturbationSpec("gaussian_noise", 0.05, {"sigma": 0.05})]
    first = evaluate_robustness(encoder, head, [batch], specs, model_name="tiny")
    second = evaluate_robustness(encoder, head, [batch], specs, model_name="tiny")
    assert first == second and len(first) == 2
    assert first[0]["valid_classes"] == 2 and first[0]["drift"] < 1e-12
    assert abs(first[1]["delta_auroc"] - (first[1]["auroc"] - first[0]["auroc"])) < 1e-12
    assert encoder.training and all(parameter.grad is None for parameter in encoder.parameters())
    assert all(torch.equal(weights[k], value) for k, value in encoder.state_dict().items())
