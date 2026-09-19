"""Safe medical-image intensity and exact local data contracts; tiny CPU only."""
import json
import numpy as np
import pytest
import torch
from PIL import Image
from data import (IntensityConfig, LocalManifestCXRDataset, UnlabeledManifestDataset,
                  inspect_manifest, load_explicit_split)


def manifest(tmp_path, *, image="image.png", labeled=True):
    record = {"image_id": "a", "image": image}
    raw = {"samples": [record]}
    if labeled:
        raw["class_names"] = [f"c{i}" for i in range(15)]
        record.update(labels=[1] + [0]*14, boxes=[{"class_index": 0, "bbox": [0, 0, 4, 4]}])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    return path, raw


def test_uint16_refuses_silent_rgb_saturation_and_explicit_scale_preserves_levels(tmp_path):
    source = np.tile(np.linspace(0, 65535, 32, dtype=np.uint16), (32, 1))
    Image.fromarray(source).save(tmp_path / "image.png")
    path, _ = manifest(tmp_path)
    with pytest.raises(ValueError, match="8-bit L/RGB"):
        LocalManifestCXRDataset(path, tmp_path, 32)[0]
    actual = LocalManifestCXRDataset(path, tmp_path, 32,
                                    intensity=IntensityConfig("uint16_scale", 65535))[0]["image"]
    assert len(actual.unique()) == 32
    assert torch.allclose(actual[0], torch.tensor(source.astype(np.float32)/65535))
    with pytest.raises(ValueError, match="refusing silent clipping"):
        LocalManifestCXRDataset(path, tmp_path, 32, intensity={"policy": "uint16_scale", "scale": 4095})[0]


@pytest.mark.parametrize("scale", [None, 0, -1, float("nan"), float("inf"), True, 65536])
def test_uint16_requires_explicit_valid_scale(scale):
    with pytest.raises(ValueError, match="explicit finite scale"):
        IntensityConfig("uint16_scale", scale)


def test_inspection_is_metadata_only_and_unlabeled_loader_has_no_label_dependency(tmp_path, monkeypatch):
    Image.new("L", (32, 32), 128).save(tmp_path / "image.png")
    path, _ = manifest(tmp_path, labeled=False)
    original_open = Image.open
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("plan decoded an image"))
    info = inspect_manifest(path, tmp_path, labeled=False, require_files=True)
    assert info["num_samples"] == 1 and info["num_classes"] is None
    assert len(info["manifest_sha256"]) == 64 and info["image_ids"] == ["a"]
    ds = UnlabeledManifestDataset(path, tmp_path, 32)
    monkeypatch.setattr(Image, "open", original_open)
    assert set(ds[0]) == {"image", "image_id"}
    assert ds[0]["image"].shape == (3, 32, 32)


def test_plan_can_report_paths_before_external_images_are_present(tmp_path):
    path, _ = manifest(tmp_path)
    assert inspect_manifest(path, tmp_path)["num_samples"] == 1
    with pytest.raises(FileNotFoundError, match="Missing local image"):
        inspect_manifest(path, tmp_path, require_files=True)


@pytest.mark.parametrize("box", [
    {"class_index": True, "bbox": [0,0,1,1]},
    {"class_index": 0, "bbox": [0,0,float("nan"),1]},
    {"class_index": 0, "bbox": [2,0,1,1]},
    {"class_index": 1, "bbox": [0,0,1,1]},
])
def test_bad_boxes_fail_before_image_loading(tmp_path, box):
    path, raw = manifest(tmp_path)
    raw["samples"][0]["boxes"] = [box]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        inspect_manifest(path, tmp_path)


def test_explicit_historical_split_preserves_order_without_resampling(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train_ids": ["d", "a"], "test_ids": ["c", "b"]}))
    train, test = load_explicit_split(path, ["a", "b", "c", "d"], train_size=2, test_size=2)
    assert train == [3,0] and test == [2,1]
    with pytest.raises(ValueError, match="size"):
        load_explicit_split(path, ["a", "b", "c", "d"], train_size=3)


@pytest.mark.parametrize("split", [
    {"train_ids": ["a", "a"], "test_ids": ["b"]},
    {"train_ids": ["a"], "test_ids": ["a", "b"]},
    {"train_ids": ["a"], "test_ids": ["x"]},
    {"train_ids": ["a"], "test_ids": []},
    {"train_ids": ["a"], "test_ids": ["b"]},
])
def test_explicit_split_rejects_duplicate_overlap_unknown_missing_empty(tmp_path, split):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError):
        load_explicit_split(path, ["a", "b", "c"])


def test_different_ids_cannot_duplicate_one_resolved_image(tmp_path):
    path, raw = manifest(tmp_path)
    duplicate = dict(raw["samples"][0], image_id="b", image="nested/../image.png")
    raw["samples"].append(duplicate)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Duplicate resolved image path"):
        inspect_manifest(path, tmp_path)


def test_rgb16_png_cannot_hide_precision_behind_pil_rgb_mode(tmp_path):
    import struct
    import zlib
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload))
    pixels = np.full((32,32,3), 4096, dtype=">u2")
    data = b"".join(b"\0"+row.tobytes() for row in pixels)
    encoded = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32,32,16,2,0,0,0))
               + chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b""))
    (tmp_path/"image.png").write_bytes(encoded)
    path, _ = manifest(tmp_path)
    with Image.open(tmp_path/"image.png") as image:
        assert image.mode == "RGB"
    with pytest.raises(ValueError, match="8-bit L/RGB storage"):
        LocalManifestCXRDataset(path, tmp_path, 32)[0]
