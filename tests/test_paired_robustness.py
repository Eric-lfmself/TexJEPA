"""Tiny CPU regressions for paired AUROC support and invalid cosine norms."""

import math

import pytest
import torch
from torch import nn

from evaluation.robustness import PerturbationSpec, evaluate_robustness
from metrics import cosine_drift


class PixelEncoder(nn.Module):
    def forward(self, images):
        value = images[:, 0, 0, 0]
        return torch.stack([value, torch.ones_like(value)], dim=1)


def evaluate(head, labels, shift):
    images = torch.zeros(4, 1, 32, 32, device="cpu")
    images[:, 0, 0, 0] = torch.tensor([.1, .2, .3, .4], device="cpu")
    return evaluate_robustness(
        PixelEncoder().cpu(), head.cpu(),
        [{"image": images, "labels": torch.tensor(labels, device="cpu")}],
        [PerturbationSpec("brightness_shift", shift, {"shift": shift})],
        model_name="tiny")


@pytest.mark.parametrize("case", ["class_removed", "sample_removed", "same_counts_replaced"])
def test_changed_support_keeps_raw_auroc_but_marks_delta_undefined(case):
    class Head(nn.Module):
        def forward(self, features):
            value = features[:, 0]
            if case == "class_removed":
                second = torch.where(value > .43, float("nan"), -value)
                return torch.stack([value, second], dim=1)
            if case == "sample_removed":
                missing = value > .45
            else:
                missing = torch.isclose(value, value.new_tensor(.1)) | torch.isclose(value, value.new_tensor(.25))
            return torch.where(missing, float("nan"), value)[:, None]

    if case == "class_removed":
        labels, shift, expected = [[0, 0], [0, 0], [1, 1], [1, 1]], .25, (.5, 1.)
    elif case == "sample_removed":
        labels, shift, expected = [[0], [1], [0], [1]], .1, (.75, .5)
    else:
        labels, shift, expected = [[0], [0], [1], [1]], .05, (1., 1.)
    clean, changed = evaluate(Head(), labels, shift)
    assert (clean["auroc"], changed["auroc"]) == expected
    assert clean["delta_auroc"] == 0 and clean["delta_status"] == "ok"
    assert math.isnan(changed["delta_auroc"])
    assert changed["delta_status"] == "undefined_support_mismatch"
    if case == "same_counts_replaced":
        assert clean["valid_classes"] == changed["valid_classes"] == 1
        assert clean["valid_samples_per_class"] == changed["valid_samples_per_class"] == [3]


def test_score_changes_at_missing_labels_do_not_change_evaluated_support():
    class Head(nn.Module):
        def forward(self, features):
            value = features[:, 0]
            missing = torch.isclose(value, value.new_tensor(.1))
            return torch.where(missing, float("nan"), value)[:, None]

    clean, changed = evaluate(Head(), [[float("nan")], [0], [1], [1]], .05)
    assert clean["valid_samples_per_class"] == changed["valid_samples_per_class"] == [3]
    assert clean["auroc"] == changed["auroc"] == 1
    assert changed["delta_auroc"] == 0 and changed["delta_status"] == "ok"


def test_matching_finite_support_retains_perturbed_minus_clean_sign():
    class Head(nn.Module):
        def forward(self, features):
            value = features[:, 0]
            return torch.where(value < .5, value, -value)[:, None]

    clean, changed = evaluate(Head(), [[0], [0], [1], [1]], .4)
    assert clean["auroc"] == 1 and changed["auroc"] == 0
    assert changed["delta_auroc"] == -1 and changed["delta_status"] == "ok"


def test_matching_but_undefined_auroc_has_explicit_delta_status():
    class Head(nn.Module):
        def forward(self, features):
            return features[:, :1]

    rows = evaluate(Head(), [[1], [1], [1], [1]], .05)
    for row in rows:
        assert math.isnan(row["auroc"]) and math.isnan(row["delta_auroc"])
        assert row["delta_status"] == "undefined_auroc"


def test_cosine_drift_excludes_overflowed_norms_on_either_side():
    clean = torch.tensor([[1e308, 1e308], [1., 0.], [1., 0.], [0., 0.]],
                         dtype=torch.float64, device="cpu")
    changed = torch.tensor([[1., 0.], [1e308, -1e308], [1., 0.], [0., 0.]],
                           dtype=torch.float64, device="cpu")
    values = cosine_drift(clean, changed)
    assert torch.isnan(values[[0, 1, 3]]).all()
    assert values[2] == 0
