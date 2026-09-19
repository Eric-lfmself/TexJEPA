"""Convert explicit local CSV labels/boxes to the Sec. III-A manifest contract.

No image pixels are decoded. Labels must already use a caller-defined reader
aggregation and 15-class harmonization; neither is inferred from official CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from data.experiment import inspect_manifest


_DEFAULT_SCHEMA = {"image_id": "image_id", "image": "image", "box_image_id": "image_id",
                   "box_class_name": "class_name", "xmin": "xmin", "ymin": "ymin",
                   "xmax": "xmax", "ymax": "ymax"}


def _local_path(value):
    if "://" in str(value):
        raise ValueError("Only local CSV/JSON paths are accepted")
    return Path(value).resolve()


def _mapping(value, name):
    if value is None:
        return {}
    raw = value if isinstance(value, dict) else json.loads(_local_path(value).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a JSON object")
    return raw


def _rows(path, *, allow_empty=False):
    source = _local_path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or any(not name or not name.strip() for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV requires unique nonempty column headers")
        rows = list(reader)
    if not rows and not allow_empty:
        raise ValueError("CSV has no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row width does not match its header")
    return source, set(columns), rows


def _cell(row, column):
    value = row[column].strip()
    if not value:
        raise ValueError(f"CSV column {column!r} contains an empty value")
    return value


def prepare_manifest(images_csv, image_root, output, *, class_map=None, boxes_csv=None,
                     unlabeled=False, schema=None, require_files=True, overwrite=False):
    """Validate the complete conversion before atomically publishing JSON.

    Class map: {class_names: [15 names], label_columns: {name: CSV column}}.
    Optional schema renames canonical image/box columns; omitted keys retain
    canonical names. Binary labels are literal 0/1; uncertainty is not guessed.
    """
    if Path(output).is_symlink():
        raise ValueError("Output must not be a symlink")
    destination = _local_path(output)
    if destination.is_symlink() or (destination.exists() and (not overwrite or not destination.is_file())):
        raise ValueError("Output exists or is a symlink; use overwrite only for a regular manifest file")
    renames = _mapping(schema, "schema")
    if set(renames) - set(_DEFAULT_SCHEMA):
        raise ValueError("Unknown CSV schema key")
    columns = {**_DEFAULT_SCHEMA, **renames}
    if any(not isinstance(name, str) or not name.strip() for name in columns.values()):
        raise ValueError("CSV schema column names must be nonempty strings")
    mapping = _mapping(class_map, "class_map")
    if unlabeled:
        if class_map is not None or boxes_csv is not None:
            raise ValueError("Unlabeled import must omit class_map and boxes_csv")
        names, label_columns = None, {}
    else:
        names, label_columns = mapping.get("class_names"), mapping.get("label_columns")
        if (not isinstance(names, list) or len(names) != 15
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != 15):
            raise ValueError("class_map requires exactly 15 explicit unique class_names")
        if (not isinstance(label_columns, dict) or set(label_columns) != set(names)
                or any(not isinstance(name, str) or not name.strip() for name in label_columns.values())
                or len(set(label_columns.values())) != 15):
            raise ValueError("label_columns must map every class to one distinct CSV column")
    source, present, image_rows = _rows(images_csv)
    required = {columns["image_id"], columns["image"], *label_columns.values()}
    if len(required) != 2 + len(label_columns):
        raise ValueError("Image ID, image path and label columns must be distinct")
    if not required <= present:
        raise ValueError(f"Images CSV missing columns: {sorted(required-present)}")
    records, by_id = [], {}
    for row in image_rows:
        image_id = _cell(row, columns["image_id"])
        if image_id in by_id:
            raise ValueError("Duplicate image_id: import requires one aggregated row per image; explicitly resolve readers first")
        record = {"image_id": image_id, "image": _cell(row, columns["image"])}
        if not unlabeled:
            labels = [_cell(row, label_columns[name]) for name in names]
            if any(value not in ("0", "1") for value in labels):
                raise ValueError("Labels must be literal binary 0/1; uncertainty/reader aggregation requires an explicit upstream policy")
            record.update(labels=[int(value) for value in labels], boxes=[])
        records.append(record)
        by_id[image_id] = record
    box_source = None
    if boxes_csv is not None:
        box_source, present, box_rows = _rows(boxes_csv, allow_empty=True)
        needed = {columns[key] for key in ("box_image_id", "box_class_name", "xmin", "ymin", "xmax", "ymax")}
        if len(needed) != 6 or not needed <= present:
            raise ValueError("Boxes CSV requires six distinct image/class/coordinate columns")
        class_index, seen = {name: index for index, name in enumerate(names)}, set()
        for row in box_rows:
            image_id, class_name = _cell(row, columns["box_image_id"]), _cell(row, columns["box_class_name"])
            if image_id not in by_id:
                raise ValueError(f"Box references unknown image_id: {image_id}")
            if class_name not in class_index:
                raise ValueError(f"Box references unknown class_name: {class_name}")
            try:
                bbox = [float(_cell(row, columns[key])) for key in ("xmin", "ymin", "xmax", "ymax")]
            except (TypeError, ValueError) as exc:
                raise ValueError("Box coordinates must be finite numeric pixel edges") from exc
            if not all(math.isfinite(value) for value in bbox) or not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3]):
                raise ValueError("Box coordinates must be finite nonempty original-image XYXY pixel edges")
            index = class_index[class_name]
            if by_id[image_id]["labels"][index] != 1:
                raise ValueError("Box class must have a positive image label")
            key = (image_id, index, *bbox)
            if key in seen:
                raise ValueError("Duplicate box; explicitly resolve duplicate reader annotations before import")
            seen.add(key)
            by_id[image_id]["boxes"].append({"class_index": index, "bbox": bbox})
    root = _local_path(image_root)
    input_paths = {source, box_source}
    input_paths.update(_local_path(value) for value in (class_map, schema) if value is not None and not isinstance(value, dict))
    if destination in input_paths or (destination.is_relative_to(root)
            and destination in {(root/record["image"]).resolve() for record in records}):
        raise ValueError("Output cannot replace an input CSV or image")
    document = {"samples": records}
    if names is not None:
        document["class_names"] = names
    document["import_provenance"] = {"images_csv": str(source), "images_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "boxes_csv": None if box_source is None else str(box_source),
        "boxes_csv_sha256": None if box_source is None else hashlib.sha256(box_source.read_bytes()).hexdigest(),
        "class_map": mapping, "schema": columns, "reader_aggregation": "already_explicit_in_input_not_inferred",
        "image_pixels_decoded": False}
    encoded = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Validate labels, box geometry, path confinement and duplicate image paths
    # using the same metadata contract as actual training, before publishing.
    with tempfile.TemporaryDirectory(prefix="texjepa-manifest-check-") as temporary:
        candidate = Path(temporary)/"manifest.json"
        candidate.write_text(encoded)
        info = inspect_manifest(candidate, root, labeled=not unlabeled, require_files=require_files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck to avoid replacing a file that appeared while CSVs validated.
        if destination.is_symlink() or (destination.exists() and (not overwrite or not destination.is_file())):
            raise ValueError("Output appeared during conversion; refusing replacement")
        if overwrite:
            os.replace(pending, destination)
        else:
            # Atomic create-if-absent: a concurrently-created output is retained.
            os.link(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
    return {"output": str(destination), "num_samples": info["num_samples"], "num_classes": info["num_classes"],
            "manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "image_pixels_decoded": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--boxes-csv", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true", help="Validate metadata without requiring each image file to exist")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_manifest(args.images_csv, args.image_root, args.output, class_map=args.class_map,
        boxes_csv=args.boxes_csv, unlabeled=args.unlabeled, schema=args.schema,
        require_files=not args.allow_missing_images, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
