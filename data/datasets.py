"""Sec III-A data contract: image, 15 binary labels, image_id and lesion boxes.

Images are CHW float32 in [0,1]. Boxes are exclusive-right/bottom XYXY pixel
coordinates. They are metadata for Sec IV-D only. No download/API functionality
is provided. Source image IDs are sorted before the seeded split; reproducing
historical split membership requires the original ID manifest.
"""
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypedDict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .experiment import inspect_manifest


class LesionBox(TypedDict):
    class_index: int
    bbox: list[float]


class CXRSample(TypedDict):
    image: torch.Tensor
    labels: torch.Tensor
    image_id: str
    boxes: list[LesionBox]


class DummyCXRDataset(Dataset):
    """Index-seeded random examples; no files, network access or shared RNG state."""
    def __init__(self, size=8, image_size=32, num_classes=15, seed=42):
        if size < 1 or image_size not in (32, 64) or num_classes != 15:
            raise ValueError("Dummy data needs positive size, 32/64 images and 15 classes")
        self.size, self.image_size, self.num_classes, self.seed = size, image_size, num_classes, seed
        self.image_ids = [f"dummy-{i:06d}" for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        g = torch.Generator(device="cpu").manual_seed(self.seed + index * 104729)
        image = torch.rand(3, self.image_size, self.image_size, generator=g, device="cpu")
        labels = (torch.rand(self.num_classes, generator=g, device="cpu") < 0.35).float()
        boxes = []
        for _ in range(index % 3):
            cls = int(torch.randint(self.num_classes, (1,), generator=g))
            side = self.image_size // 4
            x = int(torch.randint(self.image_size - side + 1, (1,), generator=g))
            y = int(torch.randint(self.image_size - side + 1, (1,), generator=g))
            labels[cls] = 1
            boxes.append({"class_index": cls, "bbox": [x, y, x + side, y + side]})
        return {"image": image, "labels": labels, "image_id": self.image_ids[index], "boxes": boxes}

    def labels_tensor(self):
        return torch.stack([self[i]["labels"] for i in range(len(self))])


@dataclass(frozen=True)
class IntensityConfig:
    """Explicit storage scaling; never infer a clinical window from one image.

    uint8 accepts only PIL L/RGB 8-bit images. uint16_scale accepts integer
    grayscale images and divides by an explicitly supplied acquisition scale.
    Values outside [0, scale] raise instead of silently clipping.
    """
    policy: str = "uint8"
    scale: float | None = None

    def __post_init__(self):
        if self.policy not in ("uint8", "uint16_scale"):
            raise ValueError("Intensity policy must be uint8 or uint16_scale")
        if self.policy == "uint8" and self.scale is not None:
            raise ValueError("uint8 has fixed scale 255; omit scale")
        if self.policy == "uint16_scale" and (type(self.scale) not in (int, float)
                or not math.isfinite(self.scale) or not 0 < self.scale <= 65535):
            raise ValueError("uint16_scale requires an explicit finite scale in (0,65535]")


def _intensity(value):
    if value is None:
        return IntensityConfig()
    if isinstance(value, dict):
        return IntensityConfig(**value)
    if not isinstance(value, IntensityConfig):
        raise ValueError("intensity must be IntensityConfig or its configuration dict")
    return value


def _read_image(path, image_size, intensity):
    with Image.open(path) as source:
        width, height = source.size
        if source.format not in ("PNG", "JPEG"):
            raise ValueError("Local image contract accepts PNG/JPEG only; convert other formats explicitly")
        storage_depth = 8
        if source.format == "PNG":
            # PIL exposes 16-bit RGB PNG as mode RGB after implicit truncation.
            # Inspect IHDR storage precision before any pixel conversion.
            with Path(path).open("rb") as handle:
                header = handle.read(25)
            if len(header) != 25 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError("Invalid PNG header")
            storage_depth = header[24]
        if intensity.policy == "uint8":
            if storage_depth != 8:
                raise ValueError("uint8 policy requires 8-bit L/RGB storage; specify an explicit intensity policy")
            if source.mode not in ("L", "RGB"):
                raise ValueError(f"uint8 policy accepts only 8-bit L/RGB images, got {source.mode}; specify an explicit intensity policy")
            resized = source.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            array = np.array(resized, dtype=np.float32) / 255.0
        else:
            if source.mode not in ("I;16", "I;16L", "I;16B", "I"):
                raise ValueError(f"uint16_scale requires integer grayscale input, got {source.mode}")
            raw = np.array(source, dtype=np.float32)
            if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > intensity.scale:
                raise ValueError("Image pixels exceed the explicit uint16 intensity scale; refusing silent clipping")
            # Resize normalized float grayscale; RGB conversion of I;16 would saturate.
            floating = Image.fromarray(raw / intensity.scale)
            plane = np.array(floating.resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32)
            array = np.repeat(plane[:, :, None], 3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1), width, height


class LocalManifestCXRDataset(Dataset):
    """Local labeled JSON with explicit 15-class mapping (Sec. III-A/IV-D).

    Bounding boxes refer to original image pixel edges and are scaled on resize.
    Use inspect_manifest for metadata-only planning without image decoding.
    DICOM interpretation, polarity and clinical windowing must be performed by
    an explicit upstream conversion; no arbitrary per-image normalization occurs.
    """
    def __init__(self, manifest, image_root, image_size=224, *, intensity=None, require_files=True):
        if type(image_size) is not int or image_size < 1:
            raise ValueError("image_size must be a positive integer")
        self.intensity = _intensity(intensity)
        self.manifest_info = inspect_manifest(manifest, image_root, labeled=True, require_files=require_files)
        self.class_names = self.manifest_info["class_names"]
        self.records = self.manifest_info["samples"]
        self.image_ids = self.manifest_info["image_ids"]
        self.root, self.image_size = Path(self.manifest_info["image_root"]), image_size

    def __len__(self):
        return len(self.records)

    def labels_tensor(self):
        return torch.tensor([r["labels"] for r in self.records], dtype=torch.float32, device="cpu")

    def __getitem__(self, index):
        record = self.records[index]
        image, width, height = _read_image(self.root / record["image"], self.image_size, self.intensity)
        boxes = []
        for box in record.get("boxes", []):
            x1, y1, x2, y2 = box["bbox"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Bounding box lies outside the source image")
            boxes.append({"class_index": box["class_index"], "bbox": [
                x1*self.image_size/width, y1*self.image_size/height,
                x2*self.image_size/width, y2*self.image_size/height]})
        return {"image": image, "labels": torch.tensor(record["labels"], dtype=torch.float32),
                "image_id": record["image_id"], "boxes": boxes}


class UnlabeledManifestDataset(Dataset):
    """Local MIMIC/pretraining records with image IDs and paths only (Fig.1/IV-E)."""
    def __init__(self, manifest, image_root, image_size=224, *, intensity=None, require_files=True):
        if type(image_size) is not int or image_size < 1:
            raise ValueError("image_size must be a positive integer")
        self.intensity = _intensity(intensity)
        self.manifest_info = inspect_manifest(manifest, image_root, labeled=False, require_files=require_files)
        self.records, self.image_ids = self.manifest_info["samples"], self.manifest_info["image_ids"]
        self.root, self.image_size = Path(self.manifest_info["image_root"]), image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, _, _ = _read_image(self.root / record["image"], self.image_size, self.intensity)
        return {"image": image, "image_id": record["image_id"]}


def collate_cxr(samples):
    """Stack tensors while preserving variable-length boxes separately."""
    return {"image": torch.stack([s["image"] for s in samples]),
            "labels": torch.stack([s["labels"] for s in samples]),
            "image_id": [s["image_id"] for s in samples],
            "boxes": [s["boxes"] for s in samples]}


def deterministic_split(image_ids, train_size=12000, test_size=3000, seed=42):
    """Return disjoint original indices, stable to input listing order.

    This is an image-level split specified by the manuscript; no unverified
    claim of patient-level independence or historical index equality is made.
    """
    if train_size < 1 or test_size < 1 or train_size + test_size != len(image_ids):
        raise ValueError("Positive train/test sizes must sum to the manifest length")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image IDs cannot be split safely")
    ordered = sorted(range(len(image_ids)), key=image_ids.__getitem__)
    order = torch.randperm(len(ordered), generator=torch.Generator(device="cpu").manual_seed(seed)).tolist()
    shuffled = [ordered[i] for i in order]
    return shuffled[:train_size], shuffled[train_size:]
