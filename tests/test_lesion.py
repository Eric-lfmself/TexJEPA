"""Numerical CPU tests of Eq. 3, all-lesion exclusion and image clustering."""

import math

import torch
from torch import nn

from evaluation.lesion_occlusion import (boxes_overlap, cluster_bootstrap_ci,
                                         evaluate_lesion_occlusion, sample_control_boxes)


class RegionEncoder(nn.Module):
    def forward(self, x):
        return x[:, :, 0:8, 0:8].mean(dim=(1, 2, 3)).unsqueeze(1)


def test_controls_exact_area_exclude_every_class():
    lesions = [(0, 0, 8, 8), (16, 0, 32, 16)]
    controls = sample_control_boxes((1, 32, 32), lesions[0], lesions,
                                    generator=torch.Generator().manual_seed(42))
    assert len(controls) == 5
    for box in controls:
        assert (box[2] - box[0]) * (box[3] - box[1]) == 64
        assert all(not boxes_overlap(box, lesion) for lesion in lesions)
    assert sample_control_boxes((1, 32, 32), (0, 0, 32, 32), [(0, 0, 32, 32)]) == []


def test_lesion_delta_uses_target_class_and_pairs_not_boxes():
    encoder, head = RegionEncoder().cpu(), nn.Linear(1, 2, bias=False).cpu()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2.], [-3.]]))
    samples = []
    for image_index in range(2):
        image = torch.zeros(1, 32, 32, device="cpu")
        image[:, :8, :8] = 1
        samples.append({"image": image, "labels": torch.tensor([1, 1]),
                        "image_id": f"image-{image_index}", "boxes": [
                            {"class_index": 0, "bbox": [0, 0, 8, 8]},
                            {"class_index": 0, "bbox": [0, 0, 8, 8]},
                            {"class_index": 1, "bbox": [0, 0, 8, 8]}]})
    result = evaluate_lesion_occlusion(encoder, head, samples, model_name="known",
                                       bootstrap_iterations=50)
    assert len(result["per_box"]) == 6 and len(result["per_pair"]) == 4
    classes = {row["class_index"]: row for row in result["per_class"]}
    assert classes[0]["delta_lesion"] == 1.0
    assert classes[1]["delta_lesion"] == -1.5
    # Pair-weighted mean differs from box-weighted mean, preventing pseudo-replication.
    assert result["overall"]["delta_lesion"] == -0.25
    assert result["overall"]["n_images"] == 2
    assert result["overall"]["ci_low"] == result["overall"]["ci_high"] == -0.25


def test_impossible_control_is_explicit_skip():
    sample = {"image": torch.ones(1, 32, 32), "labels": torch.tensor([1]),
              "image_id": "whole", "boxes": [{"class_index": 0, "bbox": [0, 0, 32, 32]}]}
    result = evaluate_lesion_occlusion(RegionEncoder(), nn.Identity(), [sample],
                                       model_name="tiny", bootstrap_iterations=10)
    assert result["per_pair"] == []
    assert result["skipped"][0]["reason"] == "no_nonlesion_control_region"
    assert math.isnan(result["overall"]["delta_lesion"])


def test_bootstrap_resamples_image_clusters_and_marks_single_image():
    rows = [{"image_id": "a", "delta_lesion": 0.},
            {"image_id": "a", "delta_lesion": 2.},
            {"image_id": "b", "delta_lesion": 4.}]
    ci = cluster_bootstrap_ci(rows, iterations=200, seed=7)
    assert ci["mean"] == 2.0 and ci["n_images"] == 2 and ci["n_pairs"] == 3
    # Sampling image a twice retains both pairs: minimum is 1, not 0.
    assert ci["ci_low"] == 1.0 and ci["ci_high"] == 4.0
    assert ci == cluster_bootstrap_ci(rows, iterations=200, seed=7)
    insufficient = cluster_bootstrap_ci(rows[:2], iterations=10)
    assert math.isnan(insufficient["ci_low"]) and insufficient["ci_status"] == "insufficient_images"
