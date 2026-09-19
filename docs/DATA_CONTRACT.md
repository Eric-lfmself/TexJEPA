# Data manifests and splits

We use explicit JSON manifests for existing local images. The loader separates image preprocessing, label harmonization, and split membership so that each experiment records the data it actually uses. Dataset references and access pages are listed in [Sources](SOURCES.md).

## Labeled images

A labeled manifest declares 15 distinct class names in logit order. Replace the illustrative names below with the intended class mapping:

```json
{
  "class_names": ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14"],
  "samples": [
    {
      "image_id": "stable-image-id",
      "image": "relative/path/image.png",
      "labels": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "boxes": [{"class_index": 0, "bbox": [10, 20, 80, 120]}]
    }
  ]
}
```

Image paths must stay within `image_root`, including after symlink resolution. IDs are unique, and different IDs cannot refer to the same resolved file. This check does not identify re-encoded duplicate images.

Labels are binary and must already reflect the selected reader-aggregation and class-harmonization policy. The loader does not merge diagnoses, resolve uncertain labels, or infer the definition of a no-finding class. The manuscript's 15-label task therefore requires an explicit mapping; the label count alone does not identify the intended classes.

Bounding boxes use original-image pixel-edge coordinates `[x1, y1, x2, y2]`, with the right and bottom edges exclusive. Each box must correspond to a positive image-level class. Metadata validation checks finite, nonnegative, nonempty geometry; image loading checks the upper bounds and scales boxes with the resize. Boxes are used only for diagnostics.

`inspect_manifest` validates metadata without decoding pixels. With `require_files=True`, it also checks image-file existence. `LocalManifestCXRDataset` uses the same contract for execution.

## Unlabeled pretraining images

Unlabeled manifests omit class names, labels, and boxes:

```json
{
  "samples": [
    {"image_id": "stable-pretrain-id", "image": "image.png"}
  ]
}
```

`UnlabeledManifestDataset` returns only `image` and `image_id`. Use separate manifests for the unlabeled corpus and labeled downstream data.

## Intensity and geometry

The default `uint8` intensity policy accepts 8-bit grayscale or RGB images. For 16-bit grayscale PNG files, choose an explicit acquisition scale, for example:

```json
"intensity": {"policy": "uint16_scale", "scale": 65535}
```

The loader divides pixel values by the supplied scale and rejects negative or over-range intensities. A 12-bit acquisition may instead use `4095` if that matches its conversion. The loader does not infer DICOM windowing, apply modality/VOI transforms, invert MONOCHROME1 polarity, or derive a per-image scale. Complete and document those conversions before creating the image manifest.

The dataset handles resize and bounding-box geometry. The backbone adapter applies channel normalization only; its mean and standard deviation must agree with the experiment's preprocessing. Perturbations operate before backbone normalization.

## Split membership

To use a specific split, provide its image IDs:

```json
{
  "train_ids": ["train-id-1", "train-id-2"],
  "test_ids": ["test-id-1"]
}
```

Set `data.split.path` to this JSON file. The loader preserves the supplied order and requires unique, disjoint IDs whose union exactly covers the manifest, with the configured train and test sizes. Unknown IDs, omitted IDs, and size mismatches are errors.

When `data.split.path` is `null`, the implementation sorts IDs and constructs a deterministic split with the configured seed. A new seed-42 split is reproducible under this procedure, but matching the manuscript's split requires the same membership. Patient-level separation requires patient identifiers and an explicit source policy; unique image IDs alone do not establish it.

## Convert CSV metadata

For already harmonized image-level labels:

```sh
python -m scripts.prepare_manifest \
  --images-csv /srv/texjepa/images.csv \
  --image-root /srv/texjepa/images \
  --class-map /srv/texjepa/class-map.json \
  --boxes-csv /srv/texjepa/boxes.csv \
  --output /srv/texjepa/manifest.json
```

The image CSV contains one row per image, with `image_id`, `image`, and one binary column per class. Values must be literal `0` or `1`; blanks, uncertainty codes, and duplicate image rows are rejected.

The class-map JSON supplies `class_names` in logit order and a `label_columns` object mapping every class name to exactly one CSV column. For example, class `c0` could map to column `label_c0`; all 15 classes must be specified. Class names in examples are placeholders, not a disease taxonomy.

The optional boxes CSV uses:

```text
image_id,class_name,xmin,ymin,xmax,ymax
```

Coordinates are original-image pixel edges. Unknown IDs/classes, duplicate boxes, invalid geometry, and boxes attached to a negative image-level class are rejected. Omit `--boxes-csv` when only classification labels are available.

Use `--schema /srv/texjepa/schema.json` to rename canonical columns. For example:

```json
{
  "image_id": "uid",
  "image": "file",
  "box_image_id": "uid",
  "box_class_name": "disease"
}
```

Coordinate columns can also be renamed with the keys `xmin`, `ymin`, `xmax`, and `ymax`.

For unlabeled pretraining metadata:

```sh
python -m scripts.prepare_manifest \
  --images-csv /srv/texjepa/mimic-images.csv \
  --image-root /srv/texjepa/mimic \
  --unlabeled \
  --output /srv/texjepa/unlabeled-manifest.json
```

The converter reads metadata and checks file existence without decoding images. `--allow-missing-images` enables metadata preparation before image files are available. Output is written after validation; `--overwrite` explicitly permits replacing an existing manifest. Source CSVs and images cannot be overwritten. The manifest records source CSV hashes, schema, and class mapping. Split-ID JSON remains a separate input.
