r"""Compare single-box and multi-box oracle prompting for CAMRI rat MRI.

This Phase 1 experiment isolates prompt geometry. The baseline uses one oracle
box around the complete expert mask. The comparison labels the expert mask with
8-connectivity, removes components below a configurable area, runs MedSAM once
per remaining component box, and unions those independent predictions.

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
METRIC_NAMES = ("dice", "iou", "precision", "recall", "hausdorff_px")


@dataclass
class EvaluatedSlice:
    """Arrays and metadata retained only for a possible improvement overlay."""

    subject_id: str
    slice_index: int
    image_uint8: np.ndarray
    gt_mask: np.ndarray
    single_prediction: np.ndarray
    multibox_prediction: np.ndarray
    single_box: tuple[int, int, int, int]
    component_boxes: list[tuple[int, int, int, int]]
    total_components: int
    discarded_components: int
    single_dice: float
    multibox_dice: float

    @property
    def dice_change(self) -> float:
        """Return multi-box Dice minus single-box Dice."""

        return self.multibox_dice - self.single_dice


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


def component_masks_and_boxes(
    gt_mask: np.ndarray,
    min_component_area: int,
    box_margin: int,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]], int, int]:
    """Find retained 8-connected components and their margin-expanded boxes.

    Returns retained component masks, boxes, the number of discarded components,
    and the total discarded foreground area.
    """

    labels = measure.label(gt_mask.astype(bool), connectivity=2)
    component_masks: list[np.ndarray] = []
    component_boxes: list[tuple[int, int, int, int]] = []
    discarded_count = 0
    discarded_area = 0

    for region in measure.regionprops(labels):
        if region.area < min_component_area:
            discarded_count += 1
            discarded_area += int(region.area)
            continue
        component_mask = (labels == region.label).astype(np.uint8)
        box = bbox_from_mask(component_mask, margin=box_margin)
        if box is not None:
            component_masks.append(component_mask)
            component_boxes.append(box)

    return component_masks, component_boxes, discarded_count, discarded_area


def predict_multibox_union(
    model: torch.nn.Module,
    image_uint8: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    device: torch.device,
) -> np.ndarray:
    """Run one independent core MedSAM inference per box and union the masks."""

    union = np.zeros(image_uint8.shape, dtype=np.uint8)
    for box in boxes:
        prediction = medsam_official_inference(model, image_uint8, box, device)
        union = np.logical_or(union, prediction).astype(np.uint8)
    return union


def metric_columns(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix the standard core metrics for a side-by-side CSV row."""

    return {f"{prefix}_{name}": metrics[name] for name in METRIC_NAMES}


def update_example_candidates(
    candidates: list[EvaluatedSlice],
    candidate: EvaluatedSlice,
    maximum: int,
) -> None:
    """Retain only the globally largest Dice improvements in memory."""

    if maximum <= 0:
        return
    candidates.append(candidate)
    candidates.sort(key=lambda item: item.dice_change, reverse=True)
    del candidates[maximum:]


def evaluate_subject(
    pair: SubjectPair,
    model: torch.nn.Module,
    device: torch.device,
    box_margin: int,
    min_component_area: int,
    max_slices: int,
    max_examples: int,
    example_candidates: list[EvaluatedSlice],
) -> list[dict[str, object]]:
    """Evaluate identical images and metrics with the two prompt geometries."""

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
        _, component_boxes, discarded_count, discarded_area = component_masks_and_boxes(
            gt_mask,
            min_component_area=min_component_area,
            box_margin=box_margin,
        )

        single_prediction = medsam_official_inference(
            model, image_uint8, single_box, device
        )
        multibox_prediction = predict_multibox_union(
            model, image_uint8, component_boxes, device
        )
        single_metrics = segmentation_metrics(single_prediction, gt_mask)
        multibox_metrics = segmentation_metrics(multibox_prediction, gt_mask)

        row: dict[str, object] = {
            "subject_id": pair.subject_id,
            "slice_index": slice_index,
            "gt_area": int(gt_mask.sum()),
            "total_components": len(component_boxes) + discarded_count,
            "retained_components": len(component_boxes),
            "discarded_components": discarded_count,
            "discarded_component_area": discarded_area,
            "single_box_x_min": single_box[0],
            "single_box_y_min": single_box[1],
            "single_box_x_max": single_box[2],
            "single_box_y_max": single_box[3],
            "component_boxes_xyxy": ";".join(
                ",".join(str(value) for value in box) for box in component_boxes
            ),
            "single_pred_area": int(single_prediction.sum()),
            "multibox_pred_area": int(multibox_prediction.sum()),
            **metric_columns("single", single_metrics),
            **metric_columns("multibox", multibox_metrics),
            "dice_change": float(multibox_metrics["dice"] - single_metrics["dice"]),
        }
        rows.append(row)

        update_example_candidates(
            example_candidates,
            EvaluatedSlice(
                subject_id=pair.subject_id,
                slice_index=slice_index,
                image_uint8=image_uint8,
                gt_mask=gt_mask,
                single_prediction=single_prediction,
                multibox_prediction=multibox_prediction,
                single_box=single_box,
                component_boxes=component_boxes,
                total_components=len(component_boxes) + discarded_count,
                discarded_components=discarded_count,
                single_dice=float(single_metrics["dice"]),
                multibox_dice=float(multibox_metrics["dice"]),
            ),
            max_examples,
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
        }
        for metric in METRIC_NAMES:
            single = np.asarray(
                [float(row[f"single_{metric}"]) for row in subject_rows], dtype=float
            )
            multibox = np.asarray(
                [float(row[f"multibox_{metric}"]) for row in subject_rows], dtype=float
            )
            summary[f"single_mean_{metric}"] = float(np.nanmean(single))
            summary[f"multibox_mean_{metric}"] = float(np.nanmean(multibox))
            summary[f"mean_{metric}_change"] = float(np.nanmean(multibox - single))
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
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(single, multibox, s=24, alpha=0.65, color="tab:blue")
    axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Single-box Dice", ylabel="Multi-box Dice")
    axes[0].set_title("Per-slice paired Dice")
    axes[0].grid(alpha=0.2)

    labels = [str(row["subject_id"]) for row in subject_summaries]
    single_means = [float(row["single_mean_dice"]) for row in subject_summaries]
    multibox_means = [float(row["multibox_mean_dice"]) for row in subject_summaries]
    x_positions = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x_positions - width / 2, single_means, width, label="Single box")
    axes[1].bar(x_positions + width / 2, multibox_means, width, label="Multi box")
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
        multi_component = np.asarray(
            [int(row["total_components"]) > 1 for row in subject_rows], dtype=bool
        )

        figure, ax = plt.subplots(figsize=(11, 5))
        ax.plot(slice_indices, single_dice, "o-", label="Single box", linewidth=1.5)
        ax.plot(slice_indices, multibox_dice, "o-", label="Multi box", linewidth=1.5)
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

        improvement = multibox_dice - single_dice
        improved = improvement > 0
        figure, ax = plt.subplots(figsize=(11, 4.5))
        ax.axhline(0, color="black", linewidth=1)
        ax.plot(
            slice_indices,
            improvement,
            color="0.4",
            marker="o",
            linewidth=1.4,
            label="Multi Dice - single Dice",
        )
        if np.any(improved):
            ax.scatter(
                slice_indices[improved],
                improvement[improved],
                color="tab:green",
                s=48,
                zorder=4,
                label="Improvement > 0",
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


def write_presentation_summary(
    path: Path,
    rows: list[dict[str, object]],
    subject_summaries: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    """Write a concise, presentation-ready interpretation of the experiment."""

    single_dice = np.asarray([float(row["single_dice"]) for row in rows])
    multibox_dice = np.asarray([float(row["multibox_dice"]) for row in rows])
    changes = multibox_dice - single_dice
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
    conclusion = (
        "Multi-box oracle prompts improved mean Dice, supporting prompt geometry as a contributor."
        if float(np.mean(changes)) > 0
        else "Multi-box oracle prompts did not improve mean Dice in this run."
    )

    subject_lines = "\n".join(
        f"- {row['subject_id']}: {float(row['single_mean_dice']):.4f} -> "
        f"{float(row['multibox_mean_dice']):.4f} "
        f"({float(row['mean_dice_change']):+.4f})"
        for row in subject_summaries
    )
    text = f"""# MedSAM Multi-Box Oracle: Presentation Summary

## Question

Does replacing one whole-mask oracle box with independent 8-connected component
boxes improve MedSAM segmentation on difficult CAMRI slices?

## Run

- Subjects: {', '.join(str(row['subject_id']) for row in subject_summaries)}
- Non-empty slices: {len(rows)}
- Box margin: {args.box_margin} pixels
- Minimum retained component area: {args.min_component_area} pixels
- Slices with multiple GT components: {int(multi_component_gt.sum())}
- Slices receiving multiple MedSAM boxes: {int(multiple_boxes.sum())}

## Result

- Mean single-box Dice: {float(single_dice.mean()):.4f}
- Mean multi-box Dice: {float(multibox_dice.mean()):.4f}
- Mean Dice change: {float(changes.mean()):+.4f}
- Mean Dice change on multi-component GT slices: {component_change:+.4f}
- Mean Dice change on slices receiving multiple boxes: {multibox_change:+.4f}
- Improved / worsened / unchanged slices: {int((changes > 0).sum())} / {int((changes < 0).sum())} / {int((changes == 0).sum())}

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
    print(f"Output:         {args.output_dir}")
    print("Loading MedSAM checkpoint...", flush=True)
    model = load_medsam_model(args.checkpoint, device)

    all_rows: list[dict[str, object]] = []
    example_candidates: list[EvaluatedSlice] = []
    for pair in selected_pairs:
        print(f"Evaluating {pair.subject_id}...", flush=True)
        subject_rows = evaluate_subject(
            pair=pair,
            model=model,
            device=device,
            box_margin=args.box_margin,
            min_component_area=args.min_component_area,
            max_slices=args.max_slices_per_subject,
            max_examples=args.max_examples,
            example_candidates=example_candidates,
        )
        all_rows.extend(subject_rows)
        if subject_rows:
            change = np.mean([float(row["dice_change"]) for row in subject_rows])
            print(f"  Completed {len(subject_rows)} slices | mean Dice change={change:+.4f}")

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
    save_qualitative_examples(example_candidates, args.output_dir / "examples")
    save_subject_dice_plots(all_rows, args.output_dir / "figures")
    write_component_analysis(
        args.output_dir / "component_analysis.csv",
        args.output_dir / "component_analysis.md",
        all_rows,
    )

    mean_change = np.mean([float(row["dice_change"]) for row in all_rows])
    print("========== Complete ==========")
    print(f"Subjects evaluated: {len(subject_summaries)}")
    print(f"Slices evaluated:   {len(all_rows)}")
    print(f"Mean Dice change:   {mean_change:+.4f}")
    print(f"Outputs:            {args.output_dir}")


if __name__ == "__main__":
    main()
