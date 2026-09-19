"""Sec III-A: local data, metadata-only planning and exact historical split IDs."""
from .datasets import (DummyCXRDataset, LocalManifestCXRDataset, UnlabeledManifestDataset,
                       IntensityConfig, collate_cxr, deterministic_split)
from .experiment import inspect_manifest, load_explicit_split
