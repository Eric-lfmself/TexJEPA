"""TexJEPA Sec. III-A and IV-B classification and robustness metrics."""

from .classification import binary_auroc, multilabel_auroc
from .robustness import delta_auroc, cosine_drift, drift_summary

__all__ = ["binary_auroc", "multilabel_auroc", "delta_auroc", "cosine_drift", "drift_summary"]
