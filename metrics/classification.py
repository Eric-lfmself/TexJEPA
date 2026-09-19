"""One-vs-rest macro AUROC, TexJEPA Sec. III-A and Tables IV/VI.

Average ranks handle tied predictions exactly (Mann-Whitney statistic).
Single-class or empty classes are undefined (NaN); macro AUROC averages only
defined classes. Missing/nonfinite entries are excluded with counts exposed.
An entirely undefined task returns NaN and valid_classes=0, never a success.
"""

import math

import torch


def _valid_binary(target, score):
    target = torch.as_tensor(target).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    score = torch.as_tensor(score).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if target.shape != score.shape:
        raise ValueError("target and score lengths must agree")
    finite = torch.isfinite(target) & torch.isfinite(score)
    if not torch.all((target[finite] == 0) | (target[finite] == 1)):
        raise ValueError("Labels must be binary or nonfinite (missing)")
    return target[finite], score[finite]


def binary_auroc(target, score):
    y, s = _valid_binary(target, score)
    n_positive = int(y.sum())
    n_negative = y.numel() - n_positive
    if not n_positive or not n_negative:
        return float("nan")
    sorted_scores, order = torch.sort(s)
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).to(torch.float64)
    starts = ends - counts + 1
    ranks = torch.repeat_interleave((starts + ends) / 2, counts)
    positive_rank_sum = ranks[y[order] == 1].sum().item()
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def multilabel_auroc(targets, scores):
    y, s = torch.as_tensor(targets), torch.as_tensor(scores)
    if y.ndim != 2 or y.shape != s.shape or y.shape[1] < 1:
        raise ValueError("Expected equal NxC labels and scores with at least one class")
    aucs, valid_samples, positives, negatives = [], [], [], []
    for index in range(y.shape[1]):
        valid_y, valid_s = _valid_binary(y[:, index], s[:, index])
        aucs.append(binary_auroc(valid_y, valid_s))
        valid_samples.append(valid_y.numel())
        positives.append(int(valid_y.sum()))
        negatives.append(valid_y.numel() - positives[-1])
    defined = [value for value in aucs if math.isfinite(value)]
    return {"macro_auroc": sum(defined) / len(defined) if defined else float("nan"),
            "per_class_auroc": aucs, "valid_classes": len(defined),
            "total_classes": y.shape[1], "valid_samples": valid_samples,
            "positives": positives, "negatives": negatives}
