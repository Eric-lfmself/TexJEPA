"""Tiny CPU regressions for shared mitigation inputs and Gaussian equivalence."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data import collate_cxr
from evaluation.mitigation import evaluate_mitigations, gaussian_smoothing
from evaluation.robustness import PerturbationSpec, evaluate_robustness


class PixelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.child = nn.Identity()

    def forward(self, images):
        self.calls += 1
        values = images[:, 0, 0, 0]
        return torch.stack([values, torch.ones_like(values)], dim=1)


class PixelHead(nn.Module):
    def forward(self, features):
        return features[:, :1]


def _samples():
    return [{"image": torch.full((1, 32, 32), value, device="cpu"),
             "labels": torch.tensor([int(index > 1)], device="cpu"),
             "image_id": str(index), "boxes": []}
            for index, value in enumerate([.45, .48, .52, .55])]


def _specs():
    return [PerturbationSpec("gaussian_noise", .3, {"sigma": .3})]


def _without_mitigation(row):
    return {key: value for key, value in row.items()
            if key not in {"mitigation", "mitigation_parameters"}}


def test_shuffled_identity_mitigations_share_exact_scores_and_drift():
    def loader():
        return DataLoader(_samples(), batch_size=2, shuffle=True, collate_fn=collate_cxr,
                          generator=torch.Generator(device="cpu").manual_seed(0))

    expected = evaluate_robustness(PixelEncoder().cpu(), PixelHead().cpu(), loader(),
                                   _specs(), model_name="identity")
    rows = evaluate_mitigations(PixelEncoder().cpu(), PixelHead().cpu(), loader(),
                                _specs(), model_name="identity", median_kernel=1, gaussian_sigma=0)
    for name in ("none", "median", "gaussian"):
        assert [_without_mitigation(row) for row in rows if row["mitigation"] == name] == expected


def test_one_shot_stream_is_evaluated_before_next_batch_is_requested():
    encoder = PixelEncoder().cpu()
    samples = _samples()

    def stream():
        yield collate_cxr(samples[:2])
        # Eagerly caching the whole dataset would request this batch before
        # the first batch's three methods and two conditions are evaluated.
        assert encoder.calls == 6
        yield collate_cxr(samples[2:])

    rows = evaluate_mitigations(encoder, PixelHead().cpu(), stream(), _specs(),
                                model_name="stream", median_kernel=1, gaussian_sigma=0)
    assert encoder.calls == 12
    expected = [_without_mitigation(row) for row in rows[:2]]
    assert [_without_mitigation(row) for row in rows[2:4]] == expected
    assert [_without_mitigation(row) for row in rows[4:]] == expected


def test_omitting_unmodified_skips_its_forwards_and_preserves_remaining_rows():
    batches = [collate_cxr(_samples())]
    full_encoder, reduced_encoder = PixelEncoder().cpu(), PixelEncoder().cpu()
    full = evaluate_mitigations(full_encoder, PixelHead().cpu(), batches, _specs(),
                                model_name="test")
    reduced = evaluate_mitigations(reduced_encoder, PixelHead().cpu(), iter(batches), _specs(),
                                   model_name="test", include_unmodified=False)
    assert reduced == [row for row in full if row["mitigation"] != "none"]
    assert full_encoder.calls == 6 and reduced_encoder.calls == 4


def test_stream_can_reuse_its_cpu_label_buffer_without_changing_previous_targets():
    labels = torch.empty(2, 1, dtype=torch.long, device="cpu")

    def stream():
        labels.fill_(0)
        yield {"image": torch.full((2, 1, 32, 32), .2, device="cpu"), "labels": labels}
        labels.fill_(1)
        yield {"image": torch.full((2, 1, 32, 32), .8, device="cpu"), "labels": labels}

    rows = evaluate_mitigations(PixelEncoder().cpu(), PixelHead().cpu(), stream(), [],
                                model_name="reused_labels", median_kernel=1, gaussian_sigma=0)
    assert len(rows) == 3
    assert all(row["auroc"] == 1 and row["delta_auroc"] == 0 for row in rows)
    assert all(row["valid_samples_per_class"] == [4] for row in rows)


def test_shared_evaluation_restores_mixed_modes_after_preprocessing_error():
    encoder, head = PixelEncoder().cpu(), PixelHead().cpu()
    encoder.train()
    encoder.child.eval()
    head.eval()
    with pytest.raises(ValueError, match="sigma must be finite"):
        evaluate_mitigations(encoder, head, iter([collate_cxr(_samples())]), _specs(),
                             model_name="test", gaussian_sigma=float("nan"))
    assert encoder.training and not encoder.child.training and not head.training


def _dense_gaussian_reference(images, sigma, kernel_size):
    if sigma == 0:
        return images.clone()
    single = images.ndim == 3
    batch = images.unsqueeze(0) if single else images
    size = 2 * math.ceil(3 * sigma) + 1 if kernel_size is None else kernel_size
    coordinates = torch.arange(size, dtype=batch.dtype, device="cpu") - size // 2
    kernel = torch.exp(-.5 * (coordinates / sigma).square())
    kernel /= kernel.sum()
    weights = torch.outer(kernel, kernel).expand(batch.shape[1], 1, size, size)
    result = F.conv2d(F.pad(batch, (size // 2,) * 4, mode="replicate"),
                      weights, groups=batch.shape[1])
    return result.squeeze(0) if single else result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("sigma,kernel_size", [(0., None), (1., 3), (2., None)])
def test_separable_smoothing_matches_dense_reference_on_signed_images(dtype, single, sigma, kernel_size):
    images = torch.rand(2, 3, 32, 32, dtype=dtype, device="cpu",
                        generator=torch.Generator(device="cpu").manual_seed(99)) * 2 - .5
    if single:
        images = images[0]
    expected = _dense_gaussian_reference(images, sigma, kernel_size)
    actual = gaussian_smoothing(images, sigma, kernel_size)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-14
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.shape == images.shape and actual.dtype == images.dtype
