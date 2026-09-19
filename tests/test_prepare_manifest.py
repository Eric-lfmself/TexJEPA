"""Offline CSV conversion publishes only complete explicit metadata."""
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_manifest import prepare_manifest, main


def csv_file(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inputs(tmp_path):
    names = [f"c{i}" for i in range(15)]
    mapping = {"class_names": names, "label_columns": {name: f"label_{name}" for name in names}}
    row = {"image_id": "a", "image": "a.png", **{column: int(index == 0)
           for index, column in enumerate(mapping["label_columns"].values())}}
    image_csv = tmp_path/"images.csv"
    csv_file(image_csv, list(row), [row])
    # Intentionally invalid image bytes prove that conversion never decodes.
    (tmp_path/"a.png").write_bytes(b"not pixels")
    boxes = tmp_path/"boxes.csv"
    csv_file(boxes, ["image_id", "class_name", "xmin", "ymin", "xmax", "ymax"],
             [{"image_id": "a", "class_name": "c0", "xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}])
    return image_csv, mapping, boxes, row


def test_canonical_csv_to_manifest_never_reads_pixels(tmp_path, monkeypatch):
    source, mapping, boxes, _ = inputs(tmp_path)
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("Decoded pixels"))
    destination = tmp_path/"manifest.json"
    result = prepare_manifest(source, tmp_path, destination, class_map=mapping, boxes_csv=boxes)
    raw = json.loads(destination.read_text())
    assert raw["class_names"] == mapping["class_names"]
    assert raw["samples"][0]["boxes"] == [{"class_index": 0, "bbox": [1,2,3,4]}]
    assert result["num_classes"] == 15 and result["image_pixels_decoded"] is False
    assert raw["import_provenance"]["reader_aggregation"] == "already_explicit_in_input_not_inferred"


@pytest.mark.parametrize("change", ["uncertain", "duplicate", "outside", "unknown_box", "negative_box_class"])
def test_invalid_conversion_does_not_replace_existing_output(tmp_path, change):
    source, mapping, boxes, row = inputs(tmp_path)
    destination = tmp_path/"manifest.json"
    destination.write_text("previous result")
    rows = [row]
    if change == "uncertain":
        row["label_c0"] = "-1"
    elif change == "duplicate":
        rows.append(row)
    elif change == "outside":
        row["image"] = "../outside.png"
    elif change == "negative_box_class":
        row["label_c0"] = 0
    elif change == "unknown_box":
        boxes.write_text(boxes.read_text().replace("c0", "unknown"))
    csv_file(source, list(row), rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_manifest(source, tmp_path, destination, class_map=mapping, boxes_csv=boxes, overwrite=True)
    assert destination.read_text() == "previous result"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_unlabeled_and_custom_schema_cli(tmp_path, capsys):
    source = tmp_path/"unlabeled.csv"
    csv_file(source, ["uid", "file"], [{"uid": "a", "file": "missing.png"}])
    schema = tmp_path/"schema.json"
    schema.write_text(json.dumps({"image_id": "uid", "image": "file"}))
    output = tmp_path/"manifest.json"
    main(["--images-csv", str(source), "--image-root", str(tmp_path), "--output", str(output),
          "--schema", str(schema), "--unlabeled", "--allow-missing-images"])
    raw = json.loads(output.read_text())
    assert raw["samples"] == [{"image_id": "a", "image": "missing.png"}]
    assert "class_names" not in raw and json.loads(capsys.readouterr().out)["num_classes"] is None


def test_unknown_classmap_and_duplicate_columns_fail(tmp_path):
    source, mapping, boxes, _ = inputs(tmp_path)
    mapping["label_columns"]["c1"] = "label_c0"
    with pytest.raises(ValueError, match="distinct CSV column"):
        prepare_manifest(source, tmp_path, tmp_path/"out.json", class_map=mapping)
    assert not (tmp_path/"out.json").exists()


def test_output_cannot_replace_source_csv_even_with_overwrite(tmp_path):
    source, mapping, boxes, _ = inputs(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="cannot replace an input"):
        prepare_manifest(source, tmp_path, source, class_map=mapping, overwrite=True)
    assert source.read_bytes() == before


def test_empty_optional_boxes_and_classmap_source_protection(tmp_path):
    source, mapping, boxes, _ = inputs(tmp_path)
    boxes.write_text("image_id,class_name,xmin,ymin,xmax,ymax\n")
    class_path = tmp_path/"classes.json"
    class_path.write_text(json.dumps(mapping))
    before = class_path.read_bytes()
    with pytest.raises(ValueError, match="cannot replace an input"):
        prepare_manifest(source, tmp_path, class_path, class_map=class_path, boxes_csv=boxes, overwrite=True)
    assert class_path.read_bytes() == before
    target = tmp_path/"result.json"
    prepare_manifest(source, tmp_path, target, class_map=class_path, boxes_csv=boxes)
    assert json.loads(target.read_text())["samples"][0]["boxes"] == []


def test_output_symlink_and_concurrent_creation_are_not_overwritten(tmp_path, monkeypatch):
    from scripts import prepare_manifest as module
    source, mapping, _, _ = inputs(tmp_path)
    actual = tmp_path/"owner.json"
    actual.write_text("owner data")
    link = tmp_path/"alias.json"
    link.symlink_to(actual)
    with pytest.raises(ValueError, match="symlink"):
        prepare_manifest(source, tmp_path, link, class_map=mapping, overwrite=True)
    assert actual.read_text() == "owner data"
    target = tmp_path/"new.json"
    atomic_link = module.os.link
    def raced_link(pending, destination):
        Path(destination).write_text("concurrent owner")
        return atomic_link(pending, destination)
    monkeypatch.setattr(module.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        prepare_manifest(source, tmp_path, target, class_map=mapping)
    assert target.read_text() == "concurrent owner"
    assert not list(tmp_path.glob(".new.json.*.tmp"))
