"""Offline manifest inspection and exact split IDs (paper Sec. III-A).

Inspection never decodes an image. Historical split membership is supplied by
explicit IDs, not guessed from a random seed or the order of filesystem entries.
"""
import hashlib
import json
import math
from pathlib import Path


def _ids(values, label):
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{label} must be a nonempty list of nonempty string image IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate image_id in {label}")
    return values


def inspect_manifest(manifest, image_root, *, labeled=True, require_files=False):
    """Validate local JSON metadata; return records, IDs and source hash.

    `bbox` is finite original-image pixel-edge XYXY (right/bottom exclusive).
    Image decoding later checks upper bounds against the actual source size.
    Unlabeled records must omit labels/boxes to prevent accidental use as labels.
    """
    path, root = Path(manifest).resolve(), Path(image_root).resolve()
    payload = path.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list) or not raw["samples"]:
        raise ValueError("Manifest requires a nonempty samples list")
    classes = raw.get("class_names")
    if labeled:
        if (not isinstance(classes, list) or len(classes) != 15
                or any(not isinstance(x, str) or not x.strip() for x in classes)
                or len(set(classes)) != 15):
            raise ValueError("Supply an explicit, unique 15-class mapping")
    elif classes is not None:
        raise ValueError("Unlabeled manifest must omit class_names")
    records, ids = raw["samples"], []
    resolved_images = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each manifest sample must be an object")
        image_id, image = record.get("image_id"), record.get("image")
        ids.append(image_id)
        if not isinstance(image, str) or not image.strip():
            raise ValueError("Each image path must be a nonempty relative local path")
        relative = Path(image)
        resolved = (root / relative).resolve()
        if "://" in image or relative.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError("Image paths must be local and confined to image_root")
        if resolved in resolved_images:
            raise ValueError("Duplicate resolved image path under different sample IDs")
        resolved_images.add(resolved)
        if require_files and not resolved.is_file():
            raise FileNotFoundError(f"Missing local image: {resolved}")
        if not labeled:
            if "labels" in record or "boxes" in record:
                raise ValueError("Unlabeled samples must omit labels and boxes")
            continue
        labels = record.get("labels")
        if not isinstance(labels, list) or len(labels) != 15 or any(type(v) not in (int, float) or v not in (0, 1) for v in labels):
            raise ValueError("Labels must be a 15-dimensional binary vector")
        boxes = record.get("boxes", [])
        if not isinstance(boxes, list):
            raise ValueError("boxes must be a list")
        for box in boxes:
            if not isinstance(box, dict) or type(box.get("class_index")) is not int or not 0 <= box["class_index"] < 15:
                raise ValueError("Invalid class_index")
            if labels[box["class_index"]] != 1:
                raise ValueError("Annotated lesion class must have a positive label")
            coords = box.get("bbox")
            if (not isinstance(coords, (list, tuple)) or len(coords) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in coords)
                    or not (0 <= coords[0] < coords[2] and 0 <= coords[1] < coords[3])):
                raise ValueError("bbox must be finite nonempty original-image XYXY coordinates")
    _ids(ids, "manifest")
    return {"manifest_path": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "image_root": str(root), "labeled": labeled, "num_samples": len(records),
            "num_classes": 15 if labeled else None, "class_names": classes,
            "image_ids": ids, "samples": records}


def load_explicit_split(path, image_ids, *, train_size=None, test_size=None):
    """Load {train_ids: [...], test_ids: [...]} covering every manifest ID once."""
    ids = _ids(list(image_ids), "manifest")
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Split must be an object containing train_ids and test_ids")
    train, test = _ids(raw.get("train_ids"), "train_ids"), _ids(raw.get("test_ids"), "test_ids")
    if set(train) & set(test):
        raise ValueError("Train and test IDs overlap")
    supplied, expected = set(train) | set(test), set(ids)
    if supplied != expected:
        raise ValueError(f"Split must exactly cover manifest IDs: unknown={sorted(supplied-expected)}, missing={sorted(expected-supplied)}")
    for name, actual, size in (("train", len(train), train_size), ("test", len(test), test_size)):
        if size is not None and (type(size) is not int or size < 1 or actual != size):
            raise ValueError(f"{name} split size does not match required {size}")
    index = {image_id: number for number, image_id in enumerate(ids)}
    return [index[x] for x in train], [index[x] for x in test]
