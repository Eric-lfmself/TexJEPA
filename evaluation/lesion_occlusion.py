"""Classification-aligned occlusion, TexJEPA Sec. IV-D, Eq. 3, Table VIII.

Coordinates use pixel-edge XYXY (right/bottom exclusive) on the supplied
image. Neutral fill defaults to 0.5 in raw [0,1] units. Each control has the
same width and height as its lesion box, hence exactly equal pixel area, and
overlaps NO annotated lesion of ANY class. Five controls are independently
sampled with replacement: controls may overlap one another. If no valid
placement exists, that box is explicitly skipped. No fallback hides lesions.

Multiple boxes are averaged within each image-class pair before inference.
The 95% interval resamples image clusters, retaining every class-pair in the
sampled image. Box-to-pair averaging, integer rounding, exact-area controls,
and clustered bootstrap are transparent implementation choices where the
paper does not specify those details. These are not historical code claims.
"""

from collections import defaultdict
import math

import torch

from .robustness import evaluation_mode


def pixel_bbox(bbox, height, width):
    """Round outwards to pixel edges; reject out-of-image/empty annotations."""
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError("bbox must contain four finite XYXY coordinates")
    x1, y1, x2, y2 = map(float, bbox)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox must be nonempty and contained in the supplied image")
    return (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))


def boxes_overlap(a, b):
    """Boundary touching has zero area and does not constitute overlap."""
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def sample_control_boxes(image_shape, lesion_bbox, all_lesion_bboxes, *,
                         count=5, generator=None):
    """Uniform integer placements outside all lesions, or [] if impossible."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("Control count must be a positive integer")
    height, width = image_shape[-2:]
    lesion = pixel_bbox(lesion_bbox, height, width)
    rectangle_width, rectangle_height = lesion[2] - lesion[0], lesion[3] - lesion[1]
    occupied = torch.zeros(height, width, dtype=torch.int64, device="cpu")
    # Always exclude the target box too, even if a caller omitted it from list.
    for box in [lesion, *all_lesion_bboxes]:
        x1, y1, x2, y2 = pixel_bbox(box, height, width)
        occupied[y1:y2, x1:x2] = 1
    integral = torch.zeros(height + 1, width + 1, dtype=torch.int64)
    integral[1:, 1:] = occupied.cumsum(0).cumsum(1)
    overlap = (integral[rectangle_height:, rectangle_width:]
               - integral[:-rectangle_height, rectangle_width:]
               - integral[rectangle_height:, :-rectangle_width]
               + integral[:-rectangle_height, :-rectangle_width])
    candidates = torch.nonzero(overlap == 0, as_tuple=False)
    if candidates.shape[0] == 0:
        return []
    selected = torch.randint(candidates.shape[0], (count,), generator=generator)
    controls = []
    for y, x in candidates[selected].tolist():
        controls.append((x, y, x + rectangle_width, y + rectangle_height))
    return controls


def occlude(image, bbox, *, fill=0.5):
    if image.ndim != 3 or not image.is_floating_point():
        raise ValueError("Expected a floating CHW image")
    if not math.isfinite(fill) or not 0 <= fill <= 1:
        raise ValueError("Neutral fill must be finite and in [0,1]")
    x1, y1, x2, y2 = pixel_bbox(bbox, image.shape[-2], image.shape[-1])
    result = image.clone()
    result[:, y1:y2, x1:x2] = fill
    return result


def cluster_bootstrap_ci(pairs, *, value_key="delta_lesion", iterations=1000, seed=42):
    """Pair-weighted mean with a percentile 95% image-cluster bootstrap CI.

    Fewer than two independent images cannot estimate uncertainty and yields
    NaN CI endpoints with an explicit status. Input is already pair-aggregated.
    """
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    clusters = defaultdict(list)
    for row in pairs:
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("Bootstrap inputs must contain finite valid pair values")
        clusters[str(row["image_id"])].append(value)
    values = [value for group in clusters.values() for value in group]
    mean = sum(values) / len(values) if values else float("nan")
    result = {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan"),
              "n_images": len(clusters), "n_pairs": len(values),
              "bootstrap_iterations": iterations, "ci_method": "image_cluster_percentile_95",
              "ci_status": "insufficient_images" if len(clusters) < 2 else "ok"}
    if len(clusters) < 2:
        return result
    groups = list(clusters.values())
    sums = torch.tensor([sum(group) for group in groups], dtype=torch.float64)
    counts = torch.tensor([len(group) for group in groups], dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # O(images) temporary memory instead of O(iterations*images) on real data.
    estimates = torch.empty(iterations, dtype=torch.float64)
    for index in range(iterations):
        sampled = torch.randint(len(groups), (len(groups),), generator=generator)
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    result["ci_low"], result["ci_high"] = torch.quantile(
        estimates, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return result


def _summarize(pairs, *, seed, bootstrap_iterations):
    interval = cluster_bootstrap_ci(pairs, iterations=bootstrap_iterations, seed=seed)
    return {"delta_lesion": interval.pop("mean"), **interval,
            "drop_lesion": sum(row["drop_lesion"] for row in pairs) / len(pairs) if pairs else float("nan"),
            "drop_control": sum(row["drop_control"] for row in pairs) / len(pairs) if pairs else float("nan"),
            "n_boxes": sum(row["n_boxes"] for row in pairs)}


def evaluate_lesion_occlusion(encoder, head, samples, *, model_name,
                              protocol="linear", fill=0.5, controls=5,
                              bootstrap_iterations=1000, seed=42,
                              evidence="synthetic_smoke"):
    """Evaluate sample dicts with image [C,H,W], labels [K], boxes, image_id.

    Every box is {class_index: int, bbox: [x1,y1,x2,y2]}. Boxes are used only
    here, never as classification-training supervision. The class index must
    match a positive image-level label. Inputs and head outputs are checked;
    nonfinite logits fail explicitly rather than fabricating lesion scores.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    per_box, skipped, seen_ids = [], [], set()
    metadata = {"model": model_name, "protocol": protocol, "evidence": evidence}
    with evaluation_mode(encoder, head):
        for sample in samples:
            image, labels, image_id = sample["image"], sample["labels"], str(sample["image_id"])
            if image_id in seen_ids:
                raise ValueError("Each image_id must occur once; merge its class annotations first")
            seen_ids.add(image_id)
            if image.ndim != 3 or not image.is_floating_point() or labels.ndim != 1:
                raise ValueError("Expected CHW floating image and one-dimensional labels")
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Lesion occlusion expects image intensities in [0,1]")
            boxes = sample.get("boxes", [])
            if not boxes:
                skipped.append({**metadata, "image_id": image_id, "reason": "no_annotated_boxes"})
                continue
            normalized = []
            for item in boxes:
                class_index = item["class_index"]
                if not isinstance(class_index, int) or not 0 <= class_index < labels.numel():
                    raise ValueError("Box class_index must be a valid integer label index")
                if labels[class_index].item() != 1:
                    raise ValueError("Box annotation class does not match a positive image label")
                normalized.append((class_index, pixel_bbox(item["bbox"], image.shape[-2], image.shape[-1])))
            all_boxes = [box for _, box in normalized]
            clean_logits = head(encoder(image.unsqueeze(0)))
            if clean_logits.shape != (1, labels.numel()) or not torch.isfinite(clean_logits).all():
                raise ValueError("Classifier must return finite [1,num_classes] logits")
            for box_index, (class_index, box) in enumerate(normalized):
                sampled = sample_control_boxes(image.shape, box, all_boxes,
                                               count=controls, generator=generator)
                if not sampled:
                    skipped.append({**metadata, "image_id": image_id, "class_index": class_index,
                                    "box_index": box_index, "bbox": list(box),
                                    "reason": "no_nonlesion_control_region"})
                    continue
                # Small image batches even when five controls are required.
                changed = [occlude(image, rectangle, fill=fill) for rectangle in [box, *sampled]]
                logits = torch.cat([head(encoder(torch.stack(changed[start:start + 4])))
                                    for start in range(0, len(changed), 4)])
                if logits.shape != (1 + controls, labels.numel()) or not torch.isfinite(logits).all():
                    raise ValueError("Classifier must return finite occlusion logits with matching classes")
                clean_logit = clean_logits[0, class_index].item()
                lesion_drop = clean_logit - logits[0, class_index].item()
                control_drops = (clean_logits[0, class_index] - logits[1:, class_index]).cpu().tolist()
                control_drop = sum(control_drops) / controls
                per_box.append({**metadata, "image_id": image_id, "class_index": class_index,
                                "box_index": box_index, "bbox": list(box),
                                "control_boxes": [list(rectangle) for rectangle in sampled],
                                "clean_logit": clean_logit, "drop_lesion": lesion_drop,
                                "drop_control": control_drop, "control_drops": control_drops,
                                "delta_lesion": lesion_drop - control_drop, "n_controls": controls})
    grouped = defaultdict(list)
    for row in per_box:
        grouped[(row["image_id"], row["class_index"])].append(row)
    per_pair = []
    for (image_id, class_index), rows in grouped.items():
        per_pair.append({**metadata, "image_id": image_id, "class_index": class_index,
                         **{key: sum(row[key] for row in rows) / len(rows)
                            for key in ("drop_lesion", "drop_control", "delta_lesion")},
                         "n_boxes": len(rows)})
    by_image, by_class = defaultdict(list), defaultdict(list)
    for row in per_pair:
        by_image[row["image_id"]].append(row)
        by_class[row["class_index"]].append(row)
    per_image = [{**metadata, "image_id": image_id,
                  **{key: sum(row[key] for row in pairs) / len(pairs)
                     for key in ("drop_lesion", "drop_control", "delta_lesion")},
                  "n_pairs": len(pairs), "n_boxes": sum(row["n_boxes"] for row in pairs)}
                 for image_id, pairs in by_image.items()]
    per_class = [{**metadata, "class_index": class_index,
                  **_summarize(pairs, seed=seed, bootstrap_iterations=bootstrap_iterations)}
                 for class_index, pairs in sorted(by_class.items())]
    overall = {**metadata, **_summarize(per_pair, seed=seed, bootstrap_iterations=bootstrap_iterations),
               "n_input_images": len(seen_ids), "n_skipped_records": len(skipped)}
    return {"per_box": per_box, "per_pair": per_pair, "per_image": per_image,
            "per_class": per_class, "overall": overall, "skipped": skipped}
