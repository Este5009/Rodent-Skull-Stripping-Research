r"""Compare single-box and multi-box oracle prompting for CAMRI rat MRI.

This Phase 1 experiment isolates prompt geometry across a single oracle box,
fixed-margin component boxes, adaptive non-overlapping component boxes, and an
optional locally clipped adaptive union. MedSAM inference, MRI preprocessing,
and metrics are shared by every method.

The expert mask is used only to create oracle boxes and evaluate predictions.
It is never passed to MedSAM as a mask prompt or copied into a prediction.

Example difficult-subject verification:
    medsam_env\Scripts\python.exe scripts\experimental\evaluate_medsam_multibox_oracle.py \
        --subjects sub-086 --device auto
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keep Matplotlib usable during headless and sandboxed experiment runs.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TEMP", ".")) / "medsam_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.getenv("TEMP", ".")) / "medsam_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage import measure


# Reuse the validated benchmark mechanics without changing scripts/core/.
REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from evaluate_medsam_camri_rat import (  # noqa: E402
    BOX_MARGIN,
    CHECKPOINT_PATH,
    MASK_ROOT,
    MRI_ROOT,
    SubjectPair,
    bbox_from_mask,
    choose_device,
    load_medsam_model,
    load_nifti_data,
    match_subjects,
    medsam_official_inference,
    percentile_to_uint8,
    segmentation_metrics,
)


DEFAULT_SUBJECTS = ["sub-050", "sub-066", "sub-086", "sub-109", "sub-112"]
DEFAULT_OUTPUT_DIR = Path("outputs/multibox_oracle")
DEFAULT_MIN_COMPONENT_AREA = 10
DEFAULT_MAX_EXAMPLES = 10
DEFAULT_MERGE_DISTANCE = 3.0
METRIC_NAMES = ("dice", "iou", "precision", "recall", "hausdorff_px")


@dataclass
class PromptGeometry:
    """Raw and adaptive prompt construction for one expert-mask slice."""

    component_masks: list[np.ndarray]
    raw_boxes: list[tuple[int, int, int, int]]
    adaptive_masks: list[np.ndarray]
    adaptive_boxes: list[tuple[int, int, int, int]]
    detected_components: int
    discarded_components: int
    discarded_area: int
    minimum_distance_px: float
    raw_overlap_area: int
    adaptive_overlap_area: int
    components_merged: bool


@dataclass
class EvaluatedSlice:
    """Arrays and metadata retained only for a possible improvement overlay."""

    subject_id: str
    slice_index: int
    image_uint8: np.ndarray
    gt_mask: np.ndarray
    single_prediction: np.ndarray
    multibox_prediction: np.ndarray
    adaptive_prediction: np.ndarray
    clipped_prediction: np.ndarray | None
    single_box: tuple[int, int, int, int]
    component_boxes: list[tuple[int, int, int, int]]
    adaptive_boxes: list[tuple[int, int, int, int]]
    total_components: int
    discarded_components: int
    minimum_component_distance_px: float
    raw_box_overlap_area: int
    adaptive_box_overlap_area: int
    components_merged: bool
    single_dice: float
    multibox_dice: float
    adaptive_dice: float
    clipped_dice: float | None

    @property
    def dice_change(self) -> float:
        """Return multi-box Dice minus single-box Dice."""

        return self.multibox_dice - self.single_dice

    @property
    def adaptive_change(self) -> float:
        """Return the best enabled adaptive Dice minus raw multi-box Dice."""

        final_dice = self.clipped_dice if self.clipped_dice is not None else self.adaptive_dice
        return final_dice - self.multibox_dice

    @property
    def final_prediction(self) -> np.ndarray:
        """Return clipped adaptive output when enabled, otherwise adaptive output."""

        return self.clipped_prediction if self.clipped_prediction is not None else self.adaptive_prediction


def parse_args() -> argparse.Namespace:
    """Read reproducible experiment options."""

    parser = argparse.ArgumentParser(
        description="Compare single-box and multi-box oracle MedSAM prompts."
    )
    parser.add_argument("--mri-root", type=Path, default=MRI_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--box-margin", type=int, default=BOX_MARGIN)
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=DEFAULT_MIN_COMPONENT_AREA,
        help="Discard 8-connected expert-mask components smaller than this pixel area.",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=DEFAULT_MERGE_DISTANCE,
        help="Merge retained components at or below this minimum pixel distance.",
    )
    parser.add_argument(
        "--clip-to-component-region",
        action="store_true",
        help="Clip each adaptive prediction to its nearest-component region.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="Subject IDs to evaluate. Defaults to the five difficult subjects.",
    )
    parser.add_argument(
        "--max-slices-per-subject",
        type=int,
        default=0,
        help="Uniformly sample at most N non-empty slices per subject; 0 evaluates all.",
    )
    parser.add_argument(
        "--max-examples",
        "--max-overlays",
        dest="max_examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Save the N slices with the largest Dice improvements. Default: 10.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto uses MPS/CUDA if available, otherwise CPU.",
    )
    args = parser.parse_args()
    if args.box_margin < 0:
        parser.error("--box-margin must be non-negative")
    if args.min_component_area < 1:
        parser.error("--min-component-area must be at least 1")
    if args.merge_distance < 0:
        parser.error("--merge-distance must be non-negative")
    if args.max_slices_per_subject < 0:
        parser.error("--max-slices-per-subject must be non-negative")
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    return args


def select_subjects(
    mri_root: Path,
    mask_root: Path,
    requested_subjects: list[str],
) -> list[SubjectPair]:
    """Match data through the core implementation and retain requested IDs."""

    pairs, warnings = match_subjects(mri_root, mask_root)
    for warning in warnings:
        print(f"Data warning: {warning.subject_id}: {warning.reason}")

    pair_by_id = {pair.subject_id: pair for pair in pairs}
    selected: list[SubjectPair] = []
    missing: list[str] = []
    for subject_id in dict.fromkeys(requested_subjects):
        pair = pair_by_id.get(subject_id)
        if pair is None:
            missing.append(subject_id)
        else:
            selected.append(pair)

    if missing:
        print("Requested subjects not found: " + ", ".join(missing))
    if not selected:
        raise RuntimeError("None of the requested subjects have matched MRI and mask files.")
    return selected


def choose_slice_indices(mask_volume: np.ndarray, maximum: int) -> list[int]:
    """Return all non-empty slices, or a uniform diagnostic subset."""

    indices = [
        index
        for index in range(mask_volume.shape[2])
        if np.any(mask_volume[:, :, index] > 0)
    ]
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum)
    return sorted({indices[int(round(position))] for position in positions})


def box_intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    """Return the overlap area of two exclusive-coordinate boxes."""

    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def box_overlap_area(
    boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> int:
    """Count image pixels covered by at least two prompt boxes."""

    coverage = np.zeros(shape, dtype=np.uint16)
    for x_min, y_min, x_max, y_max in boxes:
        coverage[y_min:y_max, x_min:x_max] += 1
    return int(np.count_nonzero(coverage > 1))


def component_distance_px(first: np.ndarray, second: np.ndarray) -> float:
    """Return the exact minimum Euclidean foreground distance in pixels."""

    distances_to_first = distance_transform_edt(~first.astype(bool))
    return float(np.min(distances_to_first[second.astype(bool)]))


def extract_components(
    gt_mask: np.ndarray, min_component_area: int
) -> tuple[list[np.ndarray], int, int, int]:
    """Extract retained 8-connected masks and report filtering statistics."""

    labels = measure.label(gt_mask.astype(bool), connectivity=2)
    masks: list[np.ndarray] = []
    discarded_count = 0
    discarded_area = 0
    regions = measure.regionprops(labels)
    for region in regions:
        if region.area < min_component_area:
            discarded_count += 1
            discarded_area += int(region.area)
        else:
            masks.append((labels == region.label).astype(np.uint8))
    return masks, len(regions), discarded_count, discarded_area


def merge_nearby_components(
    component_masks: list[np.ndarray], merge_distance: float
) -> list[np.ndarray]:
    """Merge touching/nearby components and any components with overlapping tight boxes."""

    if len(component_masks) < 2:
        return [mask.copy() for mask in component_masks]

    parents = list(range(len(component_masks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    tight_boxes = [bbox_from_mask(mask, margin=0) for mask in component_masks]
    for first in range(len(component_masks)):
        for second in range(first + 1, len(component_masks)):
            first_box = tight_boxes[first]
            second_box = tight_boxes[second]
            if first_box is None or second_box is None:
                continue
            distance = component_distance_px(component_masks[first], component_masks[second])
            if distance <= merge_distance or box_intersection_area(first_box, second_box) > 0:
                union(first, second)

    grouped: dict[int, np.ndarray] = {}
    for index, mask in enumerate(component_masks):
        root = find(index)
        if root not in grouped:
            grouped[root] = np.zeros(mask.shape, dtype=np.uint8)
        grouped[root] = np.logical_or(grouped[root], mask).astype(np.uint8)
    merged_masks = list(grouped.values())

    # A transitive merge can enlarge a tight box enough to intersect another
    # group. Merge again until all tight group boxes are disjoint.
    changed = True
    while changed:
        changed = False
        for first in range(len(merged_masks)):
            first_box = bbox_from_mask(merged_masks[first], margin=0)
            if first_box is None:
                continue
            for second in range(first + 1, len(merged_masks)):
                second_box = bbox_from_mask(merged_masks[second], margin=0)
                if second_box is None:
                    continue
                if box_intersection_area(first_box, second_box) > 0:
                    merged_masks[first] = np.logical_or(
                        merged_masks[first], merged_masks[second]
                    ).astype(np.uint8)
                    del merged_masks[second]
                    changed = True
                    break
            if changed:
                break
    return merged_masks


def adaptive_nonoverlapping_boxes(
    component_masks: list[np.ndarray], margin: int
) -> list[tuple[int, int, int, int]]:
    """Expand tight boxes side-wise while preventing pairwise overlap."""

    if not component_masks:
        return []
    height, width = component_masks[0].shape
    tight_boxes = [bbox_from_mask(mask, margin=0) for mask in component_masks]
    boxes = [list(bbox_from_mask(mask, margin=margin) or (0, 0, 0, 0)) for mask in component_masks]

    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            first_tight = tight_boxes[first]
            second_tight = tight_boxes[second]
            if first_tight is None or second_tight is None:
                continue
            if box_intersection_area(tuple(boxes[first]), tuple(boxes[second])) == 0:
                continue

            separations: list[tuple[int, str, bool]] = []
            if first_tight[2] <= second_tight[0]:
                separations.append((second_tight[0] - first_tight[2], "x", True))
            elif second_tight[2] <= first_tight[0]:
                separations.append((first_tight[0] - second_tight[2], "x", False))
            if first_tight[3] <= second_tight[1]:
                separations.append((second_tight[1] - first_tight[3], "y", True))
            elif second_tight[3] <= first_tight[1]:
                separations.append((first_tight[1] - second_tight[3], "y", False))
            if not separations:
                raise RuntimeError("Adaptive tight boxes overlap after component merging.")

            # Resolve along the axis with the most available background. This
            # preserves more of the requested margin on diagonally separated boxes.
            _, axis, first_before_second = max(separations, key=lambda item: item[0])
            if axis == "x":
                if first_before_second:
                    boundary = (first_tight[2] + second_tight[0]) // 2
                    boxes[first][2] = min(boxes[first][2], boundary)
                    boxes[second][0] = max(boxes[second][0], boundary)
                else:
                    boundary = (second_tight[2] + first_tight[0]) // 2
                    boxes[second][2] = min(boxes[second][2], boundary)
                    boxes[first][0] = max(boxes[first][0], boundary)
            else:
                if first_before_second:
                    boundary = (first_tight[3] + second_tight[1]) // 2
                    boxes[first][3] = min(boxes[first][3], boundary)
                    boxes[second][1] = max(boxes[second][1], boundary)
                else:
                    boundary = (second_tight[3] + first_tight[1]) // 2
                    boxes[second][3] = min(boxes[second][3], boundary)
                    boxes[first][1] = max(boxes[first][1], boundary)

    adaptive_boxes = [
        (
            max(0, box[0]),
            max(0, box[1]),
            min(width, box[2]),
            min(height, box[3]),
        )
        for box in boxes
    ]
    if box_overlap_area(adaptive_boxes, (height, width)) != 0:
        raise RuntimeError("Adaptive box construction produced overlapping boxes.")
    return adaptive_boxes


def build_prompt_geometry(
    gt_mask: np.ndarray,
    min_component_area: int,
    box_margin: int,
    merge_distance: float,
) -> PromptGeometry:
    """Construct fixed-margin raw prompts and adaptive non-overlapping prompts."""

    component_masks, detected, discarded, discarded_area = extract_components(
        gt_mask, min_component_area
    )
    raw_boxes = [
        box
        for mask in component_masks
        if (box := bbox_from_mask(mask, margin=box_margin)) is not None
    ]
    distances = [
        component_distance_px(component_masks[first], component_masks[second])
        for first in range(len(component_masks))
        for second in range(first + 1, len(component_masks))
    ]
    adaptive_masks = merge_nearby_components(component_masks, merge_distance)
    adaptive_boxes = adaptive_nonoverlapping_boxes(adaptive_masks, box_margin)
    return PromptGeometry(
        component_masks=component_masks,
        raw_boxes=raw_boxes,
        adaptive_masks=adaptive_masks,
        adaptive_boxes=adaptive_boxes,
        detected_components=detected,
        discarded_components=discarded,
        discarded_area=discarded_area,
        minimum_distance_px=min(distances) if distances else float("nan"),
        raw_overlap_area=box_overlap_area(raw_boxes, gt_mask.shape),
        adaptive_overlap_area=box_overlap_area(adaptive_boxes, gt_mask.shape),
        components_merged=len(adaptive_masks) < len(component_masks),
    )


def nearest_component_regions(component_masks: list[np.ndarray]) -> np.ndarray:
    """Assign each image pixel to the nearest adaptive component group."""

    if not component_masks:
        raise ValueError("At least one component is required for local clipping.")
    distances = np.stack(
        [distance_transform_edt(~mask.astype(bool)) for mask in component_masks]
    )
    return np.argmin(distances, axis=0)


def predict_box_union(
    model: torch.nn.Module,
    image_uint8: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    device: torch.device,
    local_regions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run independent core inference and optionally form a locally clipped union."""

    union = np.zeros(image_uint8.shape, dtype=np.uint8)
    clipped_union = np.zeros(image_uint8.shape, dtype=np.uint8) if local_regions is not None else None
    for box_index, box in enumerate(boxes):
        prediction = medsam_official_inference(model, image_uint8, box, device)
        union = np.logical_or(union, prediction).astype(np.uint8)
        if clipped_union is not None:
            local_prediction = np.logical_and(prediction, local_regions == box_index)
            clipped_union = np.logical_or(clipped_union, local_prediction).astype(np.uint8)
    return union, clipped_union


def metric_columns(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix the standard core metrics for a side-by-side CSV row."""

    return {f"{prefix}_{name}": metrics[name] for name in METRIC_NAMES}


def optional_metric_columns(
    prefix: str, metrics: dict[str, float] | None
) -> dict[str, float | str]:
    """Return metric columns or blanks when an optional method was disabled."""

    if metrics is None:
        return {f"{prefix}_{name}": "" for name in METRIC_NAMES}
    return metric_columns(prefix, metrics)


def evaluate_subject(
    pair: SubjectPair,
    model: torch.nn.Module,
    device: torch.device,
    box_margin: int,
    min_component_area: int,
    merge_distance: float,
    clip_to_component_region: bool,
    max_slices: int,
    visual_records: list[EvaluatedSlice],
) -> list[dict[str, object]]:
    """Evaluate the baseline, raw multi-box, and adaptive prompt strategies."""

    mri = load_nifti_data(pair.mri_path)
    gt_volume = load_nifti_data(pair.mask_path) > 0
    if mri.shape != gt_volume.shape:
        raise ValueError(f"{pair.subject_id}: MRI {mri.shape} != mask {gt_volume.shape}")
    if mri.ndim != 3:
        raise ValueError(f"{pair.subject_id}: expected 3D data, got {mri.shape}")

    slice_indices = choose_slice_indices(gt_volume, max_slices)
    rows: list[dict[str, object]] = []
    for position, slice_index in enumerate(slice_indices, start=1):
        print(f"  [{position:03d}/{len(slice_indices):03d}] slice {slice_index:03d}", flush=True)
        gt_mask = gt_volume[:, :, slice_index].astype(np.uint8)
        image_uint8 = percentile_to_uint8(mri[:, :, slice_index])

        single_box = bbox_from_mask(gt_mask, margin=box_margin)
        if single_box is None:
            continue
        geometry = build_prompt_geometry(
            gt_mask,
            min_component_area=min_component_area,
            box_margin=box_margin,
            merge_distance=merge_distance,
        )

        single_prediction = medsam_official_inference(
            model, image_uint8, single_box, device
        )
        multibox_prediction, _ = predict_box_union(
            model, image_uint8, geometry.raw_boxes, device
        )
        local_regions = (
            nearest_component_regions(geometry.adaptive_masks)
            if clip_to_component_region and geometry.adaptive_masks
            else None
        )
        adaptive_prediction, clipped_prediction = predict_box_union(
            model,
            image_uint8,
            geometry.adaptive_boxes,
            device,
            local_regions=local_regions,
        )
        single_metrics = segmentation_metrics(single_prediction, gt_mask)
        multibox_metrics = segmentation_metrics(multibox_prediction, gt_mask)
        adaptive_metrics = segmentation_metrics(adaptive_prediction, gt_mask)
        clipped_metrics = (
            segmentation_metrics(clipped_prediction, gt_mask)
            if clipped_prediction is not None
            else None
        )

        row: dict[str, object] = {
            "subject_id": pair.subject_id,
            "slice_index": slice_index,
            "gt_area": int(gt_mask.sum()),
            "total_components": geometry.detected_components,
            "retained_components": len(geometry.component_masks),
            "adaptive_prompt_count": len(geometry.adaptive_boxes),
            "discarded_components": geometry.discarded_components,
            "discarded_component_area": geometry.discarded_area,
            "minimum_component_distance_px": geometry.minimum_distance_px,
            "raw_box_overlap_area": geometry.raw_overlap_area,
            "adaptive_box_overlap_area": geometry.adaptive_overlap_area,
            "components_merged": geometry.components_merged,
            "clipping_used": clipped_prediction is not None,
            "single_box_x_min": single_box[0],
            "single_box_y_min": single_box[1],
            "single_box_x_max": single_box[2],
            "single_box_y_max": single_box[3],
            "component_boxes_xyxy": ";".join(
                ",".join(str(value) for value in box) for box in geometry.raw_boxes
            ),
            "adaptive_boxes_xyxy": ";".join(
                ",".join(str(value) for value in box)
                for box in geometry.adaptive_boxes
            ),
            "single_pred_area": int(single_prediction.sum()),
            "multibox_pred_area": int(multibox_prediction.sum()),
            "adaptive_pred_area": int(adaptive_prediction.sum()),
            "adaptive_clipped_pred_area": (
                int(clipped_prediction.sum()) if clipped_prediction is not None else ""
            ),
            **metric_columns("single", single_metrics),
            **metric_columns("multibox", multibox_metrics),
            **metric_columns("adaptive", adaptive_metrics),
            **optional_metric_columns("adaptive_clipped", clipped_metrics),
            "dice_change": float(multibox_metrics["dice"] - single_metrics["dice"]),
            "adaptive_dice_change": float(
                adaptive_metrics["dice"] - multibox_metrics["dice"]
            ),
            "clipped_dice_change": (
                float(clipped_metrics["dice"] - adaptive_metrics["dice"])
                if clipped_metrics is not None
                else ""
            ),
        }
        rows.append(row)

        visual_records.append(
            EvaluatedSlice(
                subject_id=pair.subject_id,
                slice_index=slice_index,
                image_uint8=image_uint8,
                gt_mask=gt_mask,
                single_prediction=single_prediction,
                multibox_prediction=multibox_prediction,
                adaptive_prediction=adaptive_prediction,
                clipped_prediction=clipped_prediction,
                single_box=single_box,
                component_boxes=geometry.raw_boxes,
                adaptive_boxes=geometry.adaptive_boxes,
                total_components=geometry.detected_components,
                discarded_components=geometry.discarded_components,
                minimum_component_distance_px=geometry.minimum_distance_px,
                raw_box_overlap_area=geometry.raw_overlap_area,
                adaptive_box_overlap_area=geometry.adaptive_overlap_area,
                components_merged=geometry.components_merged,
                single_dice=float(single_metrics["dice"]),
                multibox_dice=float(multibox_metrics["dice"]),
                adaptive_dice=float(adaptive_metrics["dice"]),
                clipped_dice=(
                    float(clipped_metrics["dice"])
                    if clipped_metrics is not None
                    else None
                ),
            ),
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write a non-empty list of flat dictionaries to CSV."""

    rows = list(rows)
    if not rows:
        raise RuntimeError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_subjects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate the paired slice results by subject."""

    summaries: list[dict[str, object]] = []
    subject_ids = list(dict.fromkeys(str(row["subject_id"]) for row in rows))
    for subject_id in subject_ids:
        subject_rows = [row for row in rows if row["subject_id"] == subject_id]
        summary: dict[str, object] = {
            "subject_id": subject_id,
            "slices": len(subject_rows),
            "multi_component_gt_slices": sum(
                int(row["total_components"]) > 1 for row in subject_rows
            ),
            "multi_box_slices": sum(
                int(row["retained_components"]) > 1 for row in subject_rows
            ),
            "slices_improved": sum(float(row["dice_change"]) > 0 for row in subject_rows),
            "slices_worsened": sum(float(row["dice_change"]) < 0 for row in subject_rows),
            "adaptive_slices_improved": sum(
                float(row["adaptive_dice_change"]) > 0 for row in subject_rows
            ),
            "merged_slices": sum(bool(row["components_merged"]) for row in subject_rows),
            "raw_overlap_slices": sum(
                int(row["raw_box_overlap_area"]) > 0 for row in subject_rows
            ),
            "clipping_used": any(bool(row["clipping_used"]) for row in subject_rows),
        }
        for metric in METRIC_NAMES:
            single = np.asarray(
                [float(row[f"single_{metric}"]) for row in subject_rows], dtype=float
            )
            multibox = np.asarray(
                [float(row[f"multibox_{metric}"]) for row in subject_rows], dtype=float
            )
            adaptive = np.asarray(
                [float(row[f"adaptive_{metric}"]) for row in subject_rows], dtype=float
            )
            summary[f"single_mean_{metric}"] = float(np.nanmean(single))
            summary[f"multibox_mean_{metric}"] = float(np.nanmean(multibox))
            summary[f"mean_{metric}_change"] = float(np.nanmean(multibox - single))
            summary[f"adaptive_mean_{metric}"] = float(np.nanmean(adaptive))
            summary[f"adaptive_mean_{metric}_change"] = float(
                np.nanmean(adaptive - multibox)
            )
            clipped_values = [
                row[f"adaptive_clipped_{metric}"]
                for row in subject_rows
                if row[f"adaptive_clipped_{metric}"] != ""
            ]
            if clipped_values:
                clipped = np.asarray(clipped_values, dtype=float)
                matching_adaptive = np.asarray(
                    [
                        float(row[f"adaptive_{metric}"])
                        for row in subject_rows
                        if row[f"adaptive_clipped_{metric}"] != ""
                    ],
                    dtype=float,
                )
                summary[f"adaptive_clipped_mean_{metric}"] = float(
                    np.nanmean(clipped)
                )
                summary[f"clipped_mean_{metric}_change"] = float(
                    np.nanmean(clipped - matching_adaptive)
                )
            else:
                summary[f"adaptive_clipped_mean_{metric}"] = ""
                summary[f"clipped_mean_{metric}_change"] = ""
        dice_changes = np.asarray(
            [float(row["dice_change"]) for row in subject_rows], dtype=float
        )
        summary["median_dice_change"] = float(np.median(dice_changes))
        summary["largest_dice_improvement"] = float(np.max(dice_changes))
        summaries.append(summary)
    return summaries


def save_dice_figure(
    rows: list[dict[str, object]],
    subject_summaries: list[dict[str, object]],
    path: Path,
) -> None:
    """Save paired slice-level and subject-level Dice comparisons."""

    single = np.asarray([float(row["single_dice"]) for row in rows])
    multibox = np.asarray([float(row["multibox_dice"]) for row in rows])
    adaptive = np.asarray([float(row["adaptive_dice"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        single, multibox, s=24, alpha=0.6, color="tab:orange", label="Raw multi-box"
    )
    axes[0].scatter(
        single, adaptive, s=24, alpha=0.6, color="tab:green", label="Adaptive"
    )
    axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Single-box Dice", ylabel="Multi-box Dice")
    axes[0].set_title("Per-slice paired Dice")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    labels = [str(row["subject_id"]) for row in subject_summaries]
    single_means = [float(row["single_mean_dice"]) for row in subject_summaries]
    multibox_means = [float(row["multibox_mean_dice"]) for row in subject_summaries]
    adaptive_means = [float(row["adaptive_mean_dice"]) for row in subject_summaries]
    x_positions = np.arange(len(labels))
    width = 0.26
    axes[1].bar(x_positions - width, single_means, width, label="Single box")
    axes[1].bar(x_positions, multibox_means, width, label="Raw multi-box")
    axes[1].bar(x_positions + width, adaptive_means, width, label="Adaptive")
    axes[1].set_xticks(x_positions, labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_title("Mean Dice by subject")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("MedSAM oracle prompt geometry comparison")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def show_mask(ax: plt.Axes, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Display an MRI with a translucent prediction or expert-mask overlay."""

    ax.imshow(image, cmap="gray")
    overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    if np.any(mask):
        ax.contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
    ax.set_title(title)
    ax.axis("off")


def show_error_map(
    ax: plt.Axes,
    gt_mask: np.ndarray,
    prediction: np.ndarray,
    title: str,
    show_legend: bool = False,
) -> None:
    """Show true positives in gray, false positives in red, and false negatives in blue."""

    gt = gt_mask.astype(bool)
    pred = prediction.astype(bool)
    error_rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    error_rgb[np.logical_and(gt, pred)] = (0.72, 0.72, 0.72)
    error_rgb[np.logical_and(~gt, pred)] = (1.0, 0.12, 0.12)
    error_rgb[np.logical_and(gt, ~pred)] = (0.12, 0.42, 1.0)
    ax.imshow(error_rgb)
    ax.set_title(title)
    ax.axis("off")
    if show_legend:
        ax.legend(
            handles=[
                Patch(color=(1.0, 0.12, 0.12), label="False positive"),
                Patch(color=(0.12, 0.42, 1.0), label="False negative"),
                Patch(color=(0.72, 0.72, 0.72), label="True positive"),
            ],
            loc="lower left",
            fontsize=7,
            framealpha=0.85,
        )


def show_boxes(
    ax: plt.Axes,
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    title: str,
    color: str,
) -> None:
    """Draw one or more prompt boxes on the MRI slice."""

    ax.imshow(image, cmap="gray")
    for box_index, (x_min, y_min, x_max, y_max) in enumerate(boxes, start=1):
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=color,
                linewidth=1.7,
            )
        )
        if len(boxes) > 1:
            ax.text(
                x_min,
                y_min,
                str(box_index),
                color=color,
                fontsize=8,
                weight="bold",
            )
    ax.set_title(title)
    ax.axis("off")


def save_qualitative_examples(
    candidates: list[EvaluatedSlice], output_dir: Path
) -> None:
    """Save detailed comparisons for slices with the largest Dice improvements."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(candidates, start=1):
        figure, axes_grid = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes_grid.ravel()
        axes[0].imshow(item.image_uint8, cmap="gray")
        axes[0].set_title("MRI")
        axes[0].axis("off")
        show_mask(axes[1], item.image_uint8, item.gt_mask, "Expert mask")
        show_mask(
            axes[2],
            item.image_uint8,
            item.single_prediction,
            f"Single box\nDice={item.single_dice:.3f}",
        )
        show_mask(
            axes[3],
            item.image_uint8,
            item.multibox_prediction,
            f"Multi box\nDice={item.multibox_dice:.3f}",
        )
        show_error_map(
            axes[4],
            item.gt_mask,
            item.single_prediction,
            "Single-box FP / FN",
            show_legend=True,
        )
        show_error_map(
            axes[5], item.gt_mask, item.multibox_prediction, "Multi-box FP / FN"
        )
        show_boxes(
            axes[6],
            item.image_uint8,
            [item.single_box],
            "Single-box prompt",
            color="orange",
        )
        show_boxes(
            axes[7],
            item.image_uint8,
            item.component_boxes,
            f"Multi-box prompts ({len(item.component_boxes)})\n"
            f"Discarded components={item.discarded_components}",
            color="cyan",
        )

        figure.suptitle(
            f"{item.subject_id} | slice {item.slice_index} | "
            f"connected components={item.total_components} | "
            f"single Dice={item.single_dice:.3f} | "
            f"multi Dice={item.multibox_dice:.3f} | "
            f"improvement={item.dice_change:+.3f}",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        filename = (
            f"{rank:02d}_{item.subject_id}_slice-{item.slice_index:03d}_"
            f"delta-{item.dice_change:+.3f}.png"
        )
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def save_prompt_geometry_figure(
    item: EvaluatedSlice,
    path: Path,
    label: str = "",
) -> None:
    """Save a detailed prompt/prediction comparison for one evaluated slice."""

    figure, axes_grid = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes_grid.ravel()
    axes[0].imshow(item.image_uint8, cmap="gray")
    axes[0].set_title("MRI")
    axes[0].axis("off")
    show_mask(axes[1], item.image_uint8, item.gt_mask, "Expert mask")
    show_boxes(
        axes[2], item.image_uint8, [item.single_box], "Single oracle prompt", "white"
    )
    show_boxes(
        axes[3],
        item.image_uint8,
        item.component_boxes,
        f"Raw multi-box prompts\noverlap={item.raw_box_overlap_area} px",
        "orange",
    )
    show_boxes(
        axes[4],
        item.image_uint8,
        item.adaptive_boxes,
        f"Adaptive prompts\noverlap={item.adaptive_box_overlap_area} px",
        "cyan",
    )
    show_mask(
        axes[5],
        item.image_uint8,
        item.single_prediction,
        f"Single prediction\nDice={item.single_dice:.3f}",
    )
    show_mask(
        axes[6],
        item.image_uint8,
        item.multibox_prediction,
        f"Raw multi-box prediction\nDice={item.multibox_dice:.3f}",
    )
    show_mask(
        axes[7],
        item.image_uint8,
        item.adaptive_prediction,
        f"Adaptive prediction\nDice={item.adaptive_dice:.3f}",
    )
    final_name = "Adaptive + clipping" if item.clipped_prediction is not None else "Adaptive final"
    final_dice = item.clipped_dice if item.clipped_dice is not None else item.adaptive_dice
    show_mask(
        axes[8],
        item.image_uint8,
        item.final_prediction,
        f"{final_name}\nDice={final_dice:.3f}",
    )
    show_error_map(
        axes[9], item.gt_mask, item.multibox_prediction, "Raw multi-box FP / FN", True
    )
    show_error_map(
        axes[10], item.gt_mask, item.final_prediction, "Final adaptive FP / FN"
    )
    axes[11].imshow(item.image_uint8, cmap="gray")
    for box in item.component_boxes:
        x_min, y_min, x_max, y_max = box
        axes[11].add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="orange",
                linewidth=1.5,
            )
        )
    for box in item.adaptive_boxes:
        x_min, y_min, x_max, y_max = box
        axes[11].add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
        )
    axes[11].legend(
        handles=[
            Patch(color="orange", label="Raw boxes"),
            Patch(color="cyan", label="Adaptive boxes"),
        ],
        loc="lower left",
        fontsize=8,
    )
    axes[11].set_title("Prompt geometry overlay")
    axes[11].axis("off")

    distance_text = (
        f"{item.minimum_component_distance_px:.2f} px"
        if np.isfinite(item.minimum_component_distance_px)
        else "n/a"
    )
    label_prefix = f"{label} | " if label else ""
    figure.suptitle(
        f"{label_prefix}{item.subject_id} | slice {item.slice_index} | "
        f"components={item.total_components} | min distance={distance_text} | "
        f"merged={item.components_merged} | adaptive change={item.adaptive_change:+.3f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_prompt_geometry_examples(
    records: list[EvaluatedSlice], output_dir: Path, maximum: int
) -> None:
    """Save top adaptive improvements and curated prompt-geometry cases."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(records, key=lambda item: item.adaptive_change, reverse=True)
    for rank, item in enumerate(ranked[:maximum], start=1):
        filename = (
            f"{rank:02d}_{item.subject_id}_slice-{item.slice_index:03d}_"
            f"adaptive-delta-{item.adaptive_change:+.3f}.png"
        )
        save_prompt_geometry_figure(item, output_dir / filename)

    overlapping = [item for item in records if item.raw_box_overlap_area > 0]
    if overlapping:
        item = max(overlapping, key=lambda record: record.raw_box_overlap_area)
        save_prompt_geometry_figure(
            item, output_dir / "case_overlapping_raw_boxes.png", "Overlapping raw boxes"
        )
        save_prompt_geometry_figure(
            item,
            output_dir / "case_adaptive_nonoverlapping_boxes.png",
            "Adaptive non-overlapping boxes",
        )

    merged = [item for item in records if item.components_merged]
    if merged:
        item = max(merged, key=lambda record: record.adaptive_change)
        save_prompt_geometry_figure(
            item, output_dir / "case_merged_nearby_components.png", "Merged nearby components"
        )

    clipped = [item for item in records if item.clipped_prediction is not None]
    if clipped:
        item = max(
            clipped,
            key=lambda record: abs((record.clipped_dice or 0.0) - record.adaptive_dice),
        )
        save_prompt_geometry_figure(
            item, output_dir / "case_clipped_predictions.png", "Locally clipped predictions"
        )


def rows_by_subject(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group per-slice rows by subject and sort each group by slice index."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)
    for subject_rows in grouped.values():
        subject_rows.sort(key=lambda row: int(row["slice_index"]))
    return grouped


def save_subject_dice_plots(
    rows: list[dict[str, object]], output_dir: Path
) -> None:
    """Save Dice-by-slice and Dice-improvement plots for every subject."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for subject_id, subject_rows in rows_by_subject(rows).items():
        slice_indices = np.asarray(
            [int(row["slice_index"]) for row in subject_rows], dtype=int
        )
        single_dice = np.asarray(
            [float(row["single_dice"]) for row in subject_rows], dtype=float
        )
        multibox_dice = np.asarray(
            [float(row["multibox_dice"]) for row in subject_rows], dtype=float
        )
        adaptive_dice = np.asarray(
            [float(row["adaptive_dice"]) for row in subject_rows], dtype=float
        )
        multi_component = np.asarray(
            [int(row["total_components"]) > 1 for row in subject_rows], dtype=bool
        )

        figure, ax = plt.subplots(figsize=(11, 5))
        ax.plot(slice_indices, single_dice, "o-", label="Single box", linewidth=1.4)
        ax.plot(slice_indices, multibox_dice, "o-", label="Raw multi-box", linewidth=1.4)
        ax.plot(slice_indices, adaptive_dice, "o-", label="Adaptive", linewidth=1.4)
        if np.any(multi_component):
            ax.scatter(
                slice_indices[multi_component],
                multibox_dice[multi_component],
                marker="*",
                s=115,
                facecolor="gold",
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
                label="Multiple components",
            )
        ax.set(
            xlabel="Slice index",
            ylabel="Dice",
            ylim=(-0.02, 1.02),
            title=f"{subject_id}: Dice by slice",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{subject_id}_dice_by_slice.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        raw_improvement = multibox_dice - single_dice
        adaptive_improvement = adaptive_dice - multibox_dice
        improved = adaptive_improvement > 0
        figure, ax = plt.subplots(figsize=(11, 4.5))
        ax.axhline(0, color="black", linewidth=1)
        ax.plot(
            slice_indices,
            raw_improvement,
            color="tab:blue",
            marker="o",
            linewidth=1.4,
            label="Raw multi-box - single",
        )
        ax.plot(
            slice_indices,
            adaptive_improvement,
            color="tab:orange",
            marker="o",
            linewidth=1.4,
            label="Adaptive - raw multi-box",
        )
        if np.any(improved):
            ax.scatter(
                slice_indices[improved],
                adaptive_improvement[improved],
                color="tab:green",
                s=48,
                zorder=4,
                label="Adaptive improvement > 0",
            )
        clipped_values = [row["adaptive_clipped_dice"] for row in subject_rows]
        if any(value != "" for value in clipped_values):
            clipped_dice = np.asarray([float(value) for value in clipped_values])
            ax.plot(
                slice_indices,
                clipped_dice - adaptive_dice,
                color="tab:purple",
                marker=".",
                linewidth=1.2,
                label="Clipped - adaptive",
            )
        ax.set(
            xlabel="Slice index",
            ylabel="Dice improvement",
            title=f"{subject_id}: multi-box Dice improvement by slice",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{subject_id}_dice_improvement_by_slice.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_analysis_graphs(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Save aggregate prompt-geometry graphs across all evaluated slices."""

    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        ("Single", "single", "tab:blue"),
        ("Raw multi-box", "multibox", "tab:orange"),
        ("Adaptive", "adaptive", "tab:green"),
    ]
    if any(row["adaptive_clipped_dice"] != "" for row in rows):
        methods.append(("Adaptive + clipping", "adaptive_clipped", "tab:purple"))

    figure, ax = plt.subplots(figsize=(7, 6))
    for label, prefix, color in methods:
        precision = [float(row[f"{prefix}_precision"]) for row in rows]
        recall = [float(row[f"{prefix}_recall"]) for row in rows]
        ax.scatter(precision, recall, s=28, alpha=0.55, label=label, color=color)
    ax.set(
        xlabel="Precision",
        ylabel="Recall",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        title="Precision-recall by prompt strategy",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "precision_recall_all_methods.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    distances = np.asarray(
        [float(row["minimum_component_distance_px"]) for row in rows], dtype=float
    )
    adaptive_change = np.asarray(
        [float(row["adaptive_dice_change"]) for row in rows], dtype=float
    )
    finite = np.isfinite(distances)
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(distances[finite], adaptive_change[finite], alpha=0.7, color="tab:green")
    ax.set(
        xlabel="Minimum component distance (pixels)",
        ylabel="Adaptive Dice - raw multi-box Dice",
        title="Improvement vs minimum component distance",
    )
    ax.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "improvement_vs_minimum_component_distance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    component_counts = np.asarray([int(row["total_components"]) for row in rows])
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(component_counts, adaptive_change, alpha=0.5, color="tab:green")
    unique_counts = np.unique(component_counts)
    means = [adaptive_change[component_counts == count].mean() for count in unique_counts]
    ax.plot(unique_counts, means, "o-", color="black", label="Mean improvement")
    ax.set(
        xlabel="Connected components",
        ylabel="Adaptive Dice - raw multi-box Dice",
        title="Improvement vs connected-component count",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "improvement_vs_component_count.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    overlap = np.asarray([int(row["raw_box_overlap_area"]) for row in rows])
    merged = np.asarray([bool(row["components_merged"]) for row in rows])
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(
        overlap[~merged],
        adaptive_change[~merged],
        alpha=0.65,
        label="Not merged",
        color="tab:blue",
    )
    if np.any(merged):
        ax.scatter(
            overlap[merged],
            adaptive_change[merged],
            alpha=0.8,
            marker="^",
            label="Merged",
            color="tab:red",
        )
    ax.set(
        xlabel="Raw box overlap area (pixels)",
        ylabel="Adaptive Dice - raw multi-box Dice",
        title="Box overlap area vs adaptive Dice improvement",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "box_overlap_area_vs_dice_improvement.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def summarize_by_component_count(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize Dice for slices with one, two, or at least three components."""

    categories = (
        ("1 component", lambda count: count == 1),
        ("2 components", lambda count: count == 2),
        ("3+ components", lambda count: count >= 3),
    )
    summaries: list[dict[str, object]] = []
    for label, matches in categories:
        selected = [row for row in rows if matches(int(row["total_components"]))]
        if selected:
            single = np.asarray(
                [float(row["single_dice"]) for row in selected], dtype=float
            )
            multibox = np.asarray(
                [float(row["multibox_dice"]) for row in selected], dtype=float
            )
            mean_single: object = float(single.mean())
            mean_multibox: object = float(multibox.mean())
            mean_change: object = float((multibox - single).mean())
        else:
            mean_single = ""
            mean_multibox = ""
            mean_change = ""
        summaries.append(
            {
                "components": label,
                "number_of_slices": len(selected),
                "mean_single_dice": mean_single,
                "mean_multibox_dice": mean_multibox,
                "mean_dice_improvement": mean_change,
            }
        )
    return summaries


def write_component_analysis(
    csv_path: Path,
    markdown_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save the component-count Dice summary as CSV and concise Markdown."""

    summaries = summarize_by_component_count(rows)
    write_csv(csv_path, summaries)

    def format_metric(value: object, signed: bool = False) -> str:
        if value == "":
            return "n/a"
        number = float(value)
        return f"{number:+.4f}" if signed else f"{number:.4f}"

    table_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(row["components"]),
                str(row["number_of_slices"]),
                format_metric(row["mean_single_dice"]),
                format_metric(row["mean_multibox_dice"]),
                format_metric(row["mean_dice_improvement"], signed=True),
            ]
        )
        + " |"
        for row in summaries
    )
    markdown = f"""# Component Analysis

Connected components are counted in the expert mask before the minimum-area
filter is applied.

| Components | Number of slices | Mean single Dice | Mean multi Dice | Mean Dice improvement |
|---|---:|---:|---:|---:|
{table_rows}
"""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")


def write_prompt_geometry_summary(
    csv_path: Path,
    markdown_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the requested per-slice geometry and method-comparison table."""

    summary_rows: list[dict[str, object]] = []
    for row in rows:
        summary_rows.append(
            {
                "subject_id": row["subject_id"],
                "slice_index": row["slice_index"],
                "component_count": row["total_components"],
                "minimum_component_distance_px": row[
                    "minimum_component_distance_px"
                ],
                "raw_box_overlap_area": row["raw_box_overlap_area"],
                "adaptive_box_overlap_area": row["adaptive_box_overlap_area"],
                "components_merged": row["components_merged"],
                "clipping_used": row["clipping_used"],
                "single_dice": row["single_dice"],
                "raw_multibox_dice": row["multibox_dice"],
                "adaptive_dice": row["adaptive_dice"],
                "adaptive_clipped_dice": row["adaptive_clipped_dice"],
                "raw_improvement_over_single": row["dice_change"],
                "adaptive_improvement_over_raw": row["adaptive_dice_change"],
                "clipped_improvement_over_adaptive": row["clipped_dice_change"],
            }
        )
    write_csv(csv_path, summary_rows)

    def format_value(value: object, signed: bool = False) -> str:
        if value == "":
            return "n/a"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if bool(value) else "no"
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:+.3f}" if signed else f"{number:.3f}"

    table_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(row["subject_id"]),
                str(row["slice_index"]),
                str(row["component_count"]),
                format_value(row["minimum_component_distance_px"]),
                str(row["raw_box_overlap_area"]),
                str(row["adaptive_box_overlap_area"]),
                format_value(row["components_merged"]),
                format_value(row["clipping_used"]),
                format_value(row["single_dice"]),
                format_value(row["raw_multibox_dice"]),
                format_value(row["adaptive_dice"]),
                format_value(row["adaptive_clipped_dice"]),
                format_value(row["raw_improvement_over_single"], signed=True),
                format_value(row["adaptive_improvement_over_raw"], signed=True),
                format_value(
                    row["clipped_improvement_over_adaptive"], signed=True
                ),
            ]
        )
        + " |"
        for row in summary_rows
    )
    markdown = f"""# Adaptive Prompt Geometry Summary

| Subject | Slice | Components | Min distance (px) | Raw overlap | Adaptive overlap | Merged | Clipped | Single Dice | Raw Dice | Adaptive Dice | Clipped Dice | Raw - single | Adaptive - raw | Clipped - adaptive |
|---|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}
"""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")


def write_presentation_summary(
    path: Path,
    rows: list[dict[str, object]],
    subject_summaries: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    """Write a concise, presentation-ready interpretation of the experiment."""

    single_dice = np.asarray([float(row["single_dice"]) for row in rows])
    multibox_dice = np.asarray([float(row["multibox_dice"]) for row in rows])
    adaptive_dice = np.asarray([float(row["adaptive_dice"]) for row in rows])
    changes = multibox_dice - single_dice
    adaptive_changes = adaptive_dice - multibox_dice
    multi_component_gt = np.asarray([int(row["total_components"]) > 1 for row in rows])
    multiple_boxes = np.asarray([int(row["retained_components"]) > 1 for row in rows])
    component_change = (
        float(np.mean(changes[multi_component_gt]))
        if np.any(multi_component_gt)
        else float("nan")
    )
    multibox_change = (
        float(np.mean(changes[multiple_boxes])) if np.any(multiple_boxes) else float("nan")
    )
    clipped_values = [row["adaptive_clipped_dice"] for row in rows]
    clipped_dice = (
        np.asarray([float(value) for value in clipped_values], dtype=float)
        if all(value != "" for value in clipped_values)
        else None
    )
    clipped_mean_text = (
        f"{float(clipped_dice.mean()):.4f}"
        if clipped_dice is not None
        else "not evaluated"
    )
    conclusion = (
        "Adaptive prompts improved over raw multi-box prompting in this run."
        if float(np.mean(adaptive_changes)) > 0
        else "Adaptive prompts did not improve over raw multi-box prompting in this run."
    )

    subject_lines_parts: list[str] = []
    for row in subject_summaries:
        line = (
            f"- {row['subject_id']}: {float(row['single_mean_dice']):.4f} -> "
            f"{float(row['multibox_mean_dice']):.4f} -> "
            f"{float(row['adaptive_mean_dice']):.4f}"
        )
        if row["adaptive_clipped_mean_dice"] != "":
            line += f" -> {float(row['adaptive_clipped_mean_dice']):.4f} (clipped)"
        subject_lines_parts.append(line)
    subject_lines = "\n".join(subject_lines_parts)
    clipped_changes = (
        clipped_dice - adaptive_dice if clipped_dice is not None else None
    )
    text = f"""# MedSAM Multi-Box Oracle: Presentation Summary

## Question

Can adaptive non-overlapping component prompts and optional local clipping
improve on the previous fixed-margin multi-box strategy?

## Run

- Subjects: {', '.join(str(row['subject_id']) for row in subject_summaries)}
- Non-empty slices: {len(rows)}
- Box margin: {args.box_margin} pixels
- Minimum retained component area: {args.min_component_area} pixels
- Merge distance threshold: {args.merge_distance} pixels
- Local clipping enabled: {args.clip_to_component_region}
- Slices with multiple GT components: {int(multi_component_gt.sum())}
- Slices receiving multiple MedSAM boxes: {int(multiple_boxes.sum())}
- Slices with overlapping raw boxes: {sum(int(row['raw_box_overlap_area']) > 0 for row in rows)}
- Slices with merged adaptive prompts: {sum(bool(row['components_merged']) for row in rows)}

## Result

- Mean single-box Dice: {float(single_dice.mean()):.4f}
- Mean raw multi-box Dice: {float(multibox_dice.mean()):.4f}
- Mean adaptive Dice: {float(adaptive_dice.mean()):.4f}
- Mean raw improvement over single: {float(changes.mean()):+.4f}
- Mean adaptive improvement over raw: {float(adaptive_changes.mean()):+.4f}
- Mean adaptive + clipping Dice: {clipped_mean_text}
- Mean Dice change on multi-component GT slices: {component_change:+.4f}
- Mean Dice change on slices receiving multiple boxes: {multibox_change:+.4f}
- Adaptive improved / worsened / unchanged slices: {int((adaptive_changes > 0).sum())} / {int((adaptive_changes < 0).sum())} / {int((adaptive_changes == 0).sum())}
- Clipping improved / worsened / unchanged slices: {int((clipped_changes > 0).sum()) if clipped_changes is not None else 0} / {int((clipped_changes < 0).sum()) if clipped_changes is not None else 0} / {int((clipped_changes == 0).sum()) if clipped_changes is not None else 0}

{subject_lines}

## Interpretation

{conclusion} This is an oracle prompt-geometry diagnostic, not evidence that an
automatic proposal network can yet reproduce the oracle components.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run the paired oracle-prompt experiment and generate all outputs."""

    args = parse_args()
    device = choose_device(args.device)
    selected_pairs = select_subjects(args.mri_root, args.mask_root, args.subjects)

    print("========== MedSAM Multi-Box Oracle Evaluation ==========")
    print("Subjects:       " + ", ".join(pair.subject_id for pair in selected_pairs))
    print(f"Device:         {device}")
    print(f"Box margin:     {args.box_margin} px")
    print(f"Minimum area:   {args.min_component_area} px")
    print(f"Merge distance: {args.merge_distance} px")
    print(f"Local clipping: {args.clip_to_component_region}")
    print(f"Output:         {args.output_dir}")
    print("Loading MedSAM checkpoint...", flush=True)
    model = load_medsam_model(args.checkpoint, device)

    all_rows: list[dict[str, object]] = []
    visual_records: list[EvaluatedSlice] = []
    for pair in selected_pairs:
        print(f"Evaluating {pair.subject_id}...", flush=True)
        subject_rows = evaluate_subject(
            pair=pair,
            model=model,
            device=device,
            box_margin=args.box_margin,
            min_component_area=args.min_component_area,
            merge_distance=args.merge_distance,
            clip_to_component_region=args.clip_to_component_region,
            max_slices=args.max_slices_per_subject,
            visual_records=visual_records,
        )
        all_rows.extend(subject_rows)
        if subject_rows:
            raw_change = np.mean([float(row["dice_change"]) for row in subject_rows])
            adaptive_change = np.mean(
                [float(row["adaptive_dice_change"]) for row in subject_rows]
            )
            print(
                f"  Completed {len(subject_rows)} slices | "
                f"raw change={raw_change:+.4f} | adaptive change={adaptive_change:+.4f}"
            )

    if not all_rows:
        raise RuntimeError("No non-empty slices were evaluated.")

    subject_summaries = summarize_subjects(all_rows)
    write_csv(args.output_dir / "per_slice_comparison.csv", all_rows)
    write_csv(args.output_dir / "per_subject_summary.csv", subject_summaries)
    write_presentation_summary(
        args.output_dir / "presentation_summary.md",
        all_rows,
        subject_summaries,
        args,
    )
    save_dice_figure(
        all_rows,
        subject_summaries,
        args.output_dir / "dice_comparison.png",
    )
    raw_examples = sorted(
        visual_records, key=lambda item: item.dice_change, reverse=True
    )[: args.max_examples]
    save_qualitative_examples(raw_examples, args.output_dir / "examples")
    save_prompt_geometry_examples(
        visual_records,
        args.output_dir / "examples_prompt_geometry",
        args.max_examples,
    )
    save_subject_dice_plots(all_rows, args.output_dir / "figures")
    save_analysis_graphs(all_rows, args.output_dir / "figures")
    write_component_analysis(
        args.output_dir / "component_analysis.csv",
        args.output_dir / "component_analysis.md",
        all_rows,
    )
    write_prompt_geometry_summary(
        args.output_dir / "prompt_geometry_summary.csv",
        args.output_dir / "prompt_geometry_summary.md",
        all_rows,
    )

    mean_change = np.mean([float(row["dice_change"]) for row in all_rows])
    adaptive_mean_change = np.mean(
        [float(row["adaptive_dice_change"]) for row in all_rows]
    )
    print("========== Complete ==========")
    print(f"Subjects evaluated: {len(subject_summaries)}")
    print(f"Slices evaluated:   {len(all_rows)}")
    print(f"Raw Dice change:    {mean_change:+.4f}")
    print(f"Adaptive change:    {adaptive_mean_change:+.4f}")
    print(f"Outputs:            {args.output_dir}")


if __name__ == "__main__":
    main()
