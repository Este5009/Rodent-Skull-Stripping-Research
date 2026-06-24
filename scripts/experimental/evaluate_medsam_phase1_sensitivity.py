"""
Phase 1 MedSAM sensitivity sweep for CAMRI rat MRI.

Research question:
    MedSAM with oracle boxes is expected to be a favorable benchmark, yet current
    results are around 0.91 Dice. This lightweight script tests whether that
    result is sensitive to two basic choices before moving toward new models:

      1. Oracle bounding-box margin.
      2. MRI slice preprocessing before MedSAM inference.

The script intentionally writes one aggregate CSV and only a few QC overlays.
It is meant for fast debugging and hypothesis generation, not a full report.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure


# Reuse the validated CAMRI/MedSAM mechanics from scripts/core without modifying
# core code. This keeps the experimental script small and makes differences from
# the reference benchmark easier to audit.
REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from evaluate_medsam_camri_rat import (  # noqa: E402
    CHECKPOINT_PATH,
    MASK_ROOT,
    MRI_ROOT,
    bbox_from_mask,
    choose_device,
    hausdorff_distance_px,
    load_medsam_model,
    load_nifti_data,
    match_subjects,
    medsam_official_inference,
    percentile_to_uint8,
    segmentation_metrics,
)


DEFAULT_OUTPUT_DIR = Path("outputs/sensitivity")
DEFAULT_MARGINS = [0, 2, 5, 10, 15, 20]
DEFAULT_PREPROCESSING = ["percentile", "zscore", "minmax", "whitestripe_if_available"]
QC_MARGINS = [0, 10, 20]
MAX_QC_EXAMPLES = 4


@dataclass
class SliceRecord:
    """One selected 2D slice and its oracle-mask information."""

    subject_id: str
    slice_index: int
    image_2d: np.ndarray
    gt_mask: np.ndarray


@dataclass
class WhiteStripeReference:
    """Volume-level WhiteStripe-style intensity reference."""

    mean: float
    std: float
    available: bool
    reason: str = ""


def parse_args() -> argparse.Namespace:
    """Read command-line options for a quick Phase 1 sensitivity run."""

    parser = argparse.ArgumentParser(
        description="Lightweight MedSAM box-margin and preprocessing sensitivity sweep."
    )
    parser.add_argument("--mri-root", type=Path, default=MRI_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=2,
        help="Evaluate only the first N matched subjects. Default keeps the sweep quick.",
    )
    parser.add_argument(
        "--max-slices-per-subject",
        type=int,
        default=8,
        help="Uniformly sample up to N non-empty expert-mask slices per subject.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto uses MPS/CUDA if available, otherwise CPU.",
    )
    return parser.parse_args()


def finite_foreground_values(image: np.ndarray) -> np.ndarray:
    """Return finite nonzero values, falling back to all finite values."""

    finite = image[np.isfinite(image)]
    nonzero = finite[finite > 0]
    return nonzero if nonzero.size > 0 else finite


def scale_to_uint8(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip an image to a display window and scale it to MedSAM's uint8 input."""

    clipped = np.clip(values.astype(np.float32), lower, upper)
    scaled = (clipped - lower) / max(float(upper - lower), 1e-6)
    return (scaled * 255).astype(np.uint8)


def minmax_to_uint8(image_2d: np.ndarray) -> np.ndarray:
    """Convert a slice using its finite nonzero min/max intensity range."""

    foreground = finite_foreground_values(image_2d)
    if foreground.size == 0:
        return np.zeros(image_2d.shape, dtype=np.uint8)
    return scale_to_uint8(image_2d, float(foreground.min()), float(foreground.max()))


def zscore_to_uint8(image_2d: np.ndarray) -> np.ndarray:
    """Z-score a slice, then map a fixed [-3, 3] window to uint8."""

    foreground = finite_foreground_values(image_2d)
    if foreground.size == 0:
        return np.zeros(image_2d.shape, dtype=np.uint8)
    mean = float(np.mean(foreground))
    std = float(np.std(foreground))
    if std < 1e-6:
        return percentile_to_uint8(image_2d)
    z_image = (image_2d.astype(np.float32) - mean) / std
    return scale_to_uint8(z_image, -3.0, 3.0)


def estimate_whitestripe_reference(volume: np.ndarray) -> WhiteStripeReference:
    """Estimate a simple WhiteStripe-like tissue reference from one MRI volume.

    This is deliberately conservative and self-contained. It uses the dominant
    smoothed histogram peak among finite nonzero voxels as a tissue anchor. If
    peak detection is unavailable or unstable, the caller can fall back to the
    percentile preprocessing method for this exploratory branch.
    """

    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
    except ImportError as exc:
        return WhiteStripeReference(0.0, 1.0, False, f"scipy peak tools unavailable: {exc}")

    values = finite_foreground_values(volume.astype(np.float32))
    if values.size < 100:
        return WhiteStripeReference(0.0, 1.0, False, "too few foreground voxels")

    low, high = np.percentile(values, [0.5, 99.5])
    clipped = values[(values >= low) & (values <= high)]
    if clipped.size < 100:
        return WhiteStripeReference(0.0, 1.0, False, "too few clipped foreground voxels")

    counts, edges = np.histogram(clipped, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(counts.astype(np.float32), sigma=2.0)
    peaks, _ = find_peaks(
        smooth,
        prominence=max(float(smooth.max()) * 0.03, 1.0),
        distance=6,
    )
    if len(peaks) == 0:
        return WhiteStripeReference(0.0, 1.0, False, "no histogram peak found")

    peak = peaks[np.argmax(smooth[peaks])]
    peak_center = float(centers[peak])
    half_width = max(abs(peak_center) * 0.05, float(high - low) * 0.01, 1e-6)
    stripe = values[(values >= peak_center - half_width) & (values <= peak_center + half_width)]
    if stripe.size < 50:
        return WhiteStripeReference(0.0, 1.0, False, "too few WhiteStripe voxels")

    mean = float(np.mean(stripe))
    std = float(np.std(stripe))
    if std < 1e-6:
        return WhiteStripeReference(0.0, 1.0, False, "WhiteStripe standard deviation too small")
    return WhiteStripeReference(mean, std, True)


def whitestripe_to_uint8(image_2d: np.ndarray, reference: WhiteStripeReference) -> np.ndarray:
    """Apply a WhiteStripe-style z-score reference and map [-3, 3] to uint8."""

    if not reference.available:
        return percentile_to_uint8(image_2d)
    z_image = (image_2d.astype(np.float32) - reference.mean) / reference.std
    return scale_to_uint8(z_image, -3.0, 3.0)


def preprocess_slice(
    image_2d: np.ndarray,
    method: str,
    whitestripe_reference: WhiteStripeReference,
) -> np.ndarray:
    """Prepare one 2D MRI slice for MedSAM according to the requested method."""

    if method == "percentile":
        return percentile_to_uint8(image_2d)
    if method == "zscore":
        return zscore_to_uint8(image_2d)
    if method == "minmax":
        return minmax_to_uint8(image_2d)
    if method == "whitestripe_if_available":
        return whitestripe_to_uint8(image_2d, whitestripe_reference)
    raise ValueError(f"Unknown preprocessing method: {method}")


def choose_slice_indices(mask_volume: np.ndarray, max_slices: int) -> list[int]:
    """Uniformly sample non-empty mask slices for a lightweight run."""

    non_empty = [idx for idx in range(mask_volume.shape[2]) if mask_volume[:, :, idx].sum() > 0]
    if max_slices <= 0 or len(non_empty) <= max_slices:
        return non_empty

    # Uniform sampling keeps edge and central slices represented without writing
    # a large per-slice report.
    positions = np.linspace(0, len(non_empty) - 1, max_slices)
    sampled = sorted({non_empty[int(round(pos))] for pos in positions})
    return sampled


def load_selected_slices(
    mri_root: Path,
    mask_root: Path,
    max_subjects: int,
    max_slices_per_subject: int,
) -> tuple[list[SliceRecord], dict[str, WhiteStripeReference]]:
    """Load a small matched subset of subjects and slices."""

    pairs, skips = match_subjects(mri_root, mask_root)
    pairs = pairs[:max_subjects]
    if skips:
        print(f"Data warnings before truncation: {len(skips)}")
        for skip in skips[:8]:
            print(f"  - {skip.subject_id}: {skip.reason}")

    records: list[SliceRecord] = []
    whitestripe_by_subject: dict[str, WhiteStripeReference] = {}

    for pair_index, pair in enumerate(pairs, start=1):
        print(f"[{pair_index}/{len(pairs)}] Loading {pair.subject_id}")
        mri = load_nifti_data(pair.mri_path)
        gt = load_nifti_data(pair.mask_path) > 0
        if mri.shape != gt.shape:
            print(f"  Skipped shape mismatch: MRI {mri.shape} vs mask {gt.shape}")
            continue
        if mri.ndim != 3:
            print(f"  Skipped non-3D volume: {mri.shape}")
            continue

        whitestripe_by_subject[pair.subject_id] = estimate_whitestripe_reference(mri)
        selected_indices = choose_slice_indices(gt, max_slices_per_subject)
        print(f"  Selected {len(selected_indices)} non-empty slices")

        for slice_index in selected_indices:
            records.append(
                SliceRecord(
                    subject_id=pair.subject_id,
                    slice_index=slice_index,
                    image_2d=mri[:, :, slice_index].astype(np.float32),
                    gt_mask=gt[:, :, slice_index].astype(np.uint8),
                )
            )

    return records, whitestripe_by_subject


def empty_metric_lists() -> dict[str, list[float]]:
    """Create the accumulator used for one sensitivity condition."""

    return {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "hausdorff_px": [],
    }


def finite_mean(values: Iterable[float]) -> float:
    """Mean that ignores NaNs, returning NaN when no finite values exist."""

    array = np.array(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def run_sensitivity_sweep(
    records: list[SliceRecord],
    medsam_model: torch.nn.Module,
    device: torch.device,
    whitestripe_by_subject: dict[str, WhiteStripeReference],
) -> tuple[list[dict[str, object]], dict[tuple[str, int], dict[str, np.ndarray]]]:
    """Run MedSAM for each preprocessing and box-margin condition."""

    summary: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(empty_metric_lists)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    qc_predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for method in DEFAULT_PREPROCESSING:
        print(f"\nPreprocessing: {method}")
        for margin in DEFAULT_MARGINS:
            print(f"  Margin {margin}px")
            condition = (method, margin)

            for record_index, record in enumerate(records):
                reference = whitestripe_by_subject.get(
                    record.subject_id,
                    WhiteStripeReference(0.0, 1.0, False, "missing subject reference"),
                )
                image_uint8 = preprocess_slice(record.image_2d, method, reference)
                box = bbox_from_mask(record.gt_mask, margin=margin)
                if box is None:
                    continue

                pred = medsam_official_inference(medsam_model, image_uint8, box, device)
                metrics = segmentation_metrics(pred, record.gt_mask)
                summary[condition]["dice"].append(float(metrics["dice"]))
                summary[condition]["iou"].append(float(metrics["iou"]))
                summary[condition]["precision"].append(float(metrics["precision"]))
                summary[condition]["recall"].append(float(metrics["recall"]))
                summary[condition]["hausdorff_px"].append(
                    float(hausdorff_distance_px(pred, record.gt_mask))
                )
                counts[condition] += 1

                # Keep only a tiny QC cache: the first few records, one common
                # preprocessing method, and margins that show the prompt effect.
                if (
                    method == "percentile"
                    and margin in QC_MARGINS
                    and record_index < MAX_QC_EXAMPLES
                ):
                    key = (f"{record.subject_id}_slice-{record.slice_index:03d}", margin)
                    qc_predictions[key] = {
                        "image_uint8": image_uint8,
                        "gt_mask": record.gt_mask,
                        "pred_mask": pred,
                        "box": np.array(box, dtype=np.int32),
                    }

    rows: list[dict[str, object]] = []
    for method in DEFAULT_PREPROCESSING:
        for margin in DEFAULT_MARGINS:
            condition = (method, margin)
            rows.append(
                {
                    "preprocessing": method,
                    "box_margin_px": margin,
                    "n_slices": counts[condition],
                    "mean_dice": finite_mean(summary[condition]["dice"]),
                    "mean_iou": finite_mean(summary[condition]["iou"]),
                    "mean_precision": finite_mean(summary[condition]["precision"]),
                    "mean_recall": finite_mean(summary[condition]["recall"]),
                    "mean_hausdorff_px": finite_mean(summary[condition]["hausdorff_px"]),
                }
            )
    return rows, qc_predictions


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one aggregate sensitivity CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preprocessing",
        "box_margin_px",
        "n_slices",
        "mean_dice",
        "mean_iou",
        "mean_precision",
        "mean_recall",
        "mean_hausdorff_px",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_qc_figures(
    qc_dir: Path,
    qc_predictions: dict[tuple[str, int], dict[str, np.ndarray]],
) -> None:
    """Save a few side-by-side margin comparisons for visual QC."""

    qc_dir.mkdir(parents=True, exist_ok=True)
    example_ids = sorted({key[0] for key in qc_predictions})

    for example_id in example_ids:
        available = [margin for margin in QC_MARGINS if (example_id, margin) in qc_predictions]
        if not available:
            continue

        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
        if len(available) == 1:
            axes = [axes]

        for ax, margin in zip(axes, available):
            item = qc_predictions[(example_id, margin)]
            image_uint8 = item["image_uint8"]
            gt_mask = item["gt_mask"]
            pred_mask = item["pred_mask"]
            x_min, y_min, x_max, y_max = item["box"].tolist()

            ax.imshow(image_uint8, cmap="gray")
            ax.imshow(pred_mask, alpha=0.35, cmap="autumn")
            for contour in measure.find_contours(gt_mask.astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1.0)
            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=1.3,
                )
            )
            ax.set_title(f"margin {margin}px")
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(qc_dir / f"{example_id}_margin_compare.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def print_whitestripe_status(whitestripe_by_subject: dict[str, WhiteStripeReference]) -> None:
    """Report whether the optional WhiteStripe-style preprocessing was available."""

    unavailable = {
        subject_id: ref.reason
        for subject_id, ref in whitestripe_by_subject.items()
        if not ref.available
    }
    if not unavailable:
        print("WhiteStripe-style preprocessing available for selected subjects.")
        return

    print("WhiteStripe-style preprocessing fell back to percentile for:")
    for subject_id, reason in unavailable.items():
        print(f"  - {subject_id}: {reason}")


def main() -> None:
    """Run the lightweight Phase 1 sensitivity experiment."""

    args = parse_args()
    output_dir = args.output_dir
    summary_csv = output_dir / "phase1_sensitivity.csv"
    qc_dir = output_dir / "qc"

    print("========== Phase 1 MedSAM Sensitivity Sweep ==========")
    print(f"MRI root:       {args.mri_root}")
    print(f"Mask root:      {args.mask_root}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Output dir:     {output_dir}")
    print(f"Max subjects:   {args.max_subjects}")
    print(f"Max slices/sub: {args.max_slices_per_subject}")

    records, whitestripe_by_subject = load_selected_slices(
        mri_root=args.mri_root,
        mask_root=args.mask_root,
        max_subjects=args.max_subjects,
        max_slices_per_subject=args.max_slices_per_subject,
    )
    print_whitestripe_status(whitestripe_by_subject)

    if not records:
        print("No evaluable slices found. Check dataset paths and masks.")
        return

    device = choose_device(args.device)
    print(f"Using device: {device}")
    medsam_model = load_medsam_model(args.checkpoint, device)

    rows, qc_predictions = run_sensitivity_sweep(
        records=records,
        medsam_model=medsam_model,
        device=device,
        whitestripe_by_subject=whitestripe_by_subject,
    )
    write_summary_csv(summary_csv, rows)
    save_qc_figures(qc_dir, qc_predictions)

    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved QC overlays: {qc_dir}")
    print("Top rows:")
    for row in rows[:6]:
        print(
            f"  {row['preprocessing']:24s} margin={row['box_margin_px']:2d} "
            f"Dice={row['mean_dice']:.4f} IoU={row['mean_iou']:.4f}"
        )


if __name__ == "__main__":
    main()
