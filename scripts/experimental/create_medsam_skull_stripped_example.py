"""
Create one MedSAM skull-stripped CAMRI rat MRI demo volume.

This script uses the same oracle-box MedSAM inference path as the validated core
CAMRI benchmark. The expert mask is used to create a bounding-box prompt and to
measure demo metrics; the saved skull-stripped MRI is generated from MedSAM's
predicted mask:

    skull_stripped = original_mri * medsam_pred_mask

The saved NIfTI files reuse the original MRI affine and header geometry so they
can be opened directly in 3D Slicer.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

# Keep Matplotlib cache files out of the user's home directory during sandboxed
# runs and batch jobs.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "medsam_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "medsam_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from skimage import measure


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from evaluate_medsam_camri_rat import (  # noqa: E402
    CHECKPOINT_PATH,
    bbox_from_mask,
    choose_device,
    load_medsam_model,
    load_nifti_data,
    match_subjects,
    medsam_official_inference,
    percentile_to_uint8,
)


DEFAULT_MRI_ROOT = Path("../Datasets/Image_Database/CAMRI Rat Brain MRI Data")
DEFAULT_MASK_ROOT = Path("../Datasets/Mask_Database/RodentBrainMask/CAMRI Rat")
DEFAULT_OUTPUT_DIR = Path("outputs/demo_skull_stripping")
DEFAULT_SUBJECT_ID = "sub-001"
DEFAULT_BOX_MARGIN = 5
GRID_SLICES = 12


def parse_args() -> argparse.Namespace:
    """Read command-line options for the skull-stripping demo."""

    parser = argparse.ArgumentParser(
        description="Create a MedSAM oracle-box skull-stripped NIfTI demo volume."
    )
    parser.add_argument("--subject-id", default=DEFAULT_SUBJECT_ID)
    parser.add_argument("--mri-root", type=Path, default=DEFAULT_MRI_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--box-margin", type=int, default=DEFAULT_BOX_MARGIN)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto uses MPS/CUDA if available, otherwise CPU.",
    )
    return parser.parse_args()


def find_subject_pair(subject_id: str, mri_root: Path, mask_root: Path):
    """Return the matched MRI/mask pair for one subject ID."""

    pairs, skips = match_subjects(mri_root, mask_root)
    for skip in skips[:8]:
        print(f"Data warning: {skip.subject_id}: {skip.reason}")

    for pair in pairs:
        if pair.subject_id == subject_id:
            return pair

    available = ", ".join(pair.subject_id for pair in pairs[:10])
    raise ValueError(
        f"{subject_id} was not found in matched MRI/mask pairs. "
        f"First available subjects: {available}"
    )


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute Dice, IoU, precision, and recall for binary masks."""

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    return {
        "dice": 1.0 if dice_den == 0 else (2 * tp) / dice_den,
        "iou": 1.0 if iou_den == 0 else tp / iou_den,
        "precision": 1.0 if precision_den == 0 else tp / precision_den,
        "recall": 1.0 if recall_den == 0 else tp / recall_den,
    }


def non_empty_slice_indices(mask_volume: np.ndarray) -> list[int]:
    """Return all slice indices where the expert mask contains foreground."""

    return [
        slice_index
        for slice_index in range(mask_volume.shape[2])
        if mask_volume[:, :, slice_index].sum() > 0
    ]


def run_medsam_volume(
    mri_volume: np.ndarray,
    gt_mask: np.ndarray,
    medsam_model: torch.nn.Module,
    device: torch.device,
    box_margin: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Run MedSAM on every non-empty expert-mask slice and stack a 3D mask."""

    pred_mask = np.zeros(gt_mask.shape, dtype=np.uint8)
    rows: list[dict[str, object]] = []
    slice_indices = non_empty_slice_indices(gt_mask)

    for position, slice_index in enumerate(slice_indices, start=1):
        print(f"[{position}/{len(slice_indices)}] Inference slice {slice_index}")
        gt_slice = gt_mask[:, :, slice_index].astype(np.uint8)
        box = bbox_from_mask(gt_slice, margin=box_margin)
        if box is None:
            continue

        image_uint8 = percentile_to_uint8(mri_volume[:, :, slice_index])
        pred_slice = medsam_official_inference(medsam_model, image_uint8, box, device)
        pred_mask[:, :, slice_index] = pred_slice.astype(np.uint8)

        metrics = overlap_metrics(pred_slice, gt_slice)
        rows.append(
            {
                "level": "slice",
                "slice_index": slice_index,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "gt_area": int(gt_slice.sum()),
                "pred_area": int(pred_slice.sum()),
                "box_area": int((box[2] - box[0]) * (box[3] - box[1])),
            }
        )

    overall = overlap_metrics(pred_mask, gt_mask)
    rows.append(
        {
            "level": "overall",
            "slice_index": "all_non_empty_slices",
            "dice": overall["dice"],
            "iou": overall["iou"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "gt_area": int(gt_mask.sum()),
            "pred_area": int(pred_mask.sum()),
            "box_area": "",
        }
    )
    return pred_mask, rows


def save_nifti_like_original(
    path: Path,
    data: np.ndarray,
    original_img: nib.Nifti1Image,
    dtype: np.dtype,
) -> None:
    """Save a NIfTI volume with the original MRI geometry."""

    header = original_img.header.copy()
    header.set_data_dtype(dtype)
    out_img = nib.Nifti1Image(
        data.astype(dtype, copy=False),
        affine=original_img.affine,
        header=header,
    )
    # Preserve explicit qform/sform metadata when present. 3D Slicer uses these
    # fields to place the volume in physical space.
    out_img.set_qform(original_img.get_qform(), int(original_img.header["qform_code"]))
    out_img.set_sform(original_img.get_sform(), int(original_img.header["sform_code"]))
    nib.save(out_img, str(path))


def draw_boundary(ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float = 1.3) -> None:
    """Draw a binary mask boundary on an existing image axis."""

    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def save_midslice_overlay(
    path: Path,
    mri_volume: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    metrics_by_slice: dict[int, dict[str, float]],
) -> None:
    """Save one clean overlay for the middle non-empty slice."""

    slice_indices = non_empty_slice_indices(gt_mask)
    slice_index = slice_indices[len(slice_indices) // 2]
    image_uint8 = percentile_to_uint8(mri_volume[:, :, slice_index])
    dice = metrics_by_slice[slice_index]["dice"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_uint8, cmap="gray")
    draw_boundary(ax, gt_mask[:, :, slice_index], "red")
    draw_boundary(ax, pred_mask[:, :, slice_index], "lime")
    ax.set_title(f"MedSAM overlay | slice {slice_index} | Dice={dice:.3f}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def representative_slices(slice_indices: list[int], count: int) -> list[int]:
    """Choose evenly spaced representative slices from a slice list."""

    if len(slice_indices) <= count:
        return slice_indices
    positions = np.linspace(0, len(slice_indices) - 1, count)
    return sorted({slice_indices[int(round(position))] for position in positions})


def save_overlay_grid(
    path: Path,
    mri_volume: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    metrics_by_slice: dict[int, dict[str, float]],
) -> None:
    """Save representative slices with expert and MedSAM boundaries."""

    slices = representative_slices(non_empty_slice_indices(gt_mask), GRID_SLICES)
    ncols = 4
    nrows = int(np.ceil(len(slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, slice_index in zip(axes, slices):
        ax.imshow(percentile_to_uint8(mri_volume[:, :, slice_index]), cmap="gray")
        draw_boundary(ax, gt_mask[:, :, slice_index], "red", linewidth=1.0)
        draw_boundary(ax, pred_mask[:, :, slice_index], "lime", linewidth=1.0)
        ax.set_title(f"slice {slice_index} | Dice={metrics_by_slice[slice_index]['dice']:.3f}")
        ax.axis("off")

    for ax in axes[len(slices) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_skull_stripped_grid(path: Path, skull_stripped: np.ndarray, gt_mask: np.ndarray) -> None:
    """Save representative slices from the final skull-stripped MRI volume."""

    slices = representative_slices(non_empty_slice_indices(gt_mask), GRID_SLICES)
    ncols = 4
    nrows = int(np.ceil(len(slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, slice_index in zip(axes, slices):
        ax.imshow(percentile_to_uint8(skull_stripped[:, :, slice_index]), cmap="gray")
        ax.set_title(f"slice {slice_index}")
        ax.axis("off")

    for ax in axes[len(slices) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write per-slice and overall demo metrics."""

    fieldnames = [
        "level",
        "slice_index",
        "dice",
        "iou",
        "precision",
        "recall",
        "gt_area",
        "pred_area",
        "box_area",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_demo_readme(path: Path, subject_id: str) -> None:
    """Write a short README for the generated demo folder."""

    path.write_text(
        f"""# MedSAM Skull-Stripping Demo: {subject_id}

This folder contains one skull-stripping demonstration generated with MedSAM
using expert-mask oracle bounding boxes.

## Volumes

- `volumes/medsam_pred_mask.nii.gz`: 3D binary mask predicted by MedSAM.
- `volumes/medsam_skull_stripped.nii.gz`: original MRI multiplied by the MedSAM
  predicted mask.

Both NIfTI files preserve the original MRI affine/header geometry and can be
opened directly in 3D Slicer.

## Figures

- `figures/midslice_overlay.png`: middle non-empty slice with expert boundary in
  red and MedSAM boundary in green.
- `figures/slice_grid_overlay.png`: representative overlay slices across the
  expert-mask foreground extent.
- `figures/skull_stripped_slice_grid.png`: representative slices from the final
  skull-stripped MRI volume.

## Table

- `tables/demo_metrics.csv`: per-slice and overall Dice, IoU, precision, and
  recall for the MedSAM predicted mask.
""",
    )


def metrics_by_slice(rows: list[dict[str, object]]) -> dict[int, dict[str, float]]:
    """Index per-slice metrics by slice number for figure titles."""

    indexed: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["level"] != "slice":
            continue
        slice_index = int(row["slice_index"])
        indexed[slice_index] = {
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }
    return indexed


def main() -> None:
    """Create the skull-stripped demo volumes, figures, metrics, and README."""

    args = parse_args()
    subject_dir = args.output_dir / args.subject_id
    volumes_dir = subject_dir / "volumes"
    figures_dir = subject_dir / "figures"
    tables_dir = subject_dir / "tables"
    for directory in (volumes_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("========== MedSAM Skull-Stripping Demo ==========")
    print(f"Subject:      {args.subject_id}")
    print(f"MRI root:     {args.mri_root}")
    print(f"Mask root:    {args.mask_root}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Output dir:   {subject_dir}")
    print(f"Box margin:   {args.box_margin} px")

    pair = find_subject_pair(args.subject_id, args.mri_root, args.mask_root)
    original_img = nib.load(str(pair.mri_path))
    original_volume = np.asanyarray(original_img.dataobj)
    mri_volume = load_nifti_data(pair.mri_path)
    gt_mask = load_nifti_data(pair.mask_path) > 0

    if mri_volume.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: MRI {mri_volume.shape} vs mask {gt_mask.shape}")
    if mri_volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got shape {mri_volume.shape}")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    medsam_model = load_medsam_model(args.checkpoint, device)

    pred_mask, rows = run_medsam_volume(
        mri_volume=mri_volume,
        gt_mask=gt_mask,
        medsam_model=medsam_model,
        device=device,
        box_margin=args.box_margin,
    )
    skull_stripped = original_volume * pred_mask.astype(original_volume.dtype, copy=False)

    save_nifti_like_original(
        volumes_dir / "medsam_pred_mask.nii.gz",
        pred_mask,
        original_img,
        np.dtype(np.uint8),
    )
    save_nifti_like_original(
        volumes_dir / "medsam_skull_stripped.nii.gz",
        skull_stripped,
        original_img,
        original_volume.dtype,
    )

    per_slice_metrics = metrics_by_slice(rows)
    save_midslice_overlay(
        figures_dir / "midslice_overlay.png",
        mri_volume,
        gt_mask,
        pred_mask,
        per_slice_metrics,
    )
    save_overlay_grid(
        figures_dir / "slice_grid_overlay.png",
        mri_volume,
        gt_mask,
        pred_mask,
        per_slice_metrics,
    )
    save_skull_stripped_grid(
        figures_dir / "skull_stripped_slice_grid.png",
        skull_stripped,
        gt_mask,
    )
    write_metrics_csv(tables_dir / "demo_metrics.csv", rows)
    write_demo_readme(subject_dir / "README.md", args.subject_id)

    overall = rows[-1]
    print("\n========== Demo Complete ==========")
    print(f"Non-empty slices evaluated: {len(rows) - 1}")
    print(f"Overall Dice:              {float(overall['dice']):.4f}")
    print(f"Overall IoU:               {float(overall['iou']):.4f}")
    print(f"Predicted mask:            {volumes_dir / 'medsam_pred_mask.nii.gz'}")
    print(f"Skull-stripped MRI:        {volumes_dir / 'medsam_skull_stripped.nii.gz'}")


if __name__ == "__main__":
    main()
