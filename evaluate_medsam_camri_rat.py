"""
Oracle-box MedSAM benchmark for CAMRI rat brain MRI.

Research question:
    If MedSAM is given an ideal 2D bounding box derived from the expert manual
    brain mask, how accurately can it segment CAMRI rat brain MRI?

Important:
    The expert mask is used only for two things:
      1. Generate the oracle bounding box for each non-empty 2D slice.
      2. Evaluate MedSAM's predicted mask.

    The expert mask is never copied into the prediction. This keeps the
    benchmark interpretable as a best-case prompt experiment rather than a
    mask-transfer pipeline.

Example smoke test:
    MPLBACKEND=Agg medsam_env/bin/python evaluate_medsam_camri_rat.py --max-subjects 1

Full run:
    MPLBACKEND=Agg medsam_env/bin/python evaluate_medsam_camri_rat.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from skimage import measure, transform


# Keep the local MedSAM checkout importable without installing it as a package.
sys.path.append("MedSAM")
from segment_anything import sam_model_registry  # noqa: E402


# ----------------------------
# Default paths and constants
# ----------------------------
MRI_ROOT = Path("Image Database/CAMRI Rat Brain MRI Data")
MASK_ROOT = Path("Mask_Database/RodentBrainMask/CAMRI Rat")
CHECKPOINT_PATH = Path("MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
RESULTS_CSV = Path("medsam_camri_rat_results.csv")
EXAMPLE_DIR = Path("outputs/camri_rat_examples")

# Expand each oracle box by this many pixels on each side. A small margin makes
# the prompt less brittle to one-pixel mask boundary irregularities while still
# keeping the target localized.
BOX_MARGIN = 5

# Save only a few visual examples so a full run remains lightweight. The CSV is
# the complete quantitative record.
MAX_EXAMPLES = 12


@dataclass
class SubjectPair:
    """Matched MRI and expert-mask files for one subject."""

    subject_id: str
    mri_path: Path
    mask_path: Path


@dataclass
class SkipRecord:
    """A subject that could not be evaluated, plus the reason."""

    subject_id: str
    reason: str


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Most defaults match the user's requested benchmark. The extra options are
    useful for quick smoke tests and for tuning without editing the file.
    """

    # Keep all paths configurable from the command line so the same script can
    # be reused for smoke tests, full runs, and alternate CAMRI-like datasets.
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM on CAMRI rat MRI using expert-mask oracle boxes."
    )
    # The defaults point to the local workspace layout established for this
    # project. Changing them should not require editing the source file.
    parser.add_argument("--mri-root", type=Path, default=MRI_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--example-dir", type=Path, default=EXAMPLE_DIR)
    parser.add_argument("--box-margin", type=int, default=BOX_MARGIN)
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    # Limiting subjects is useful because a full oracle-box evaluation runs one
    # MedSAM pass per non-empty slice and can take several minutes on CPU.
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Evaluate only the first N matched subjects. Useful for smoke tests.",
    )
    # Device selection is explicit so failures can be reproduced; "auto" remains
    # convenient for everyday runs on Mac laptops with MPS.
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto uses MPS/CUDA if available, otherwise CPU.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """Choose an inference device, preferring Apple MPS on Mac when available."""

    # Honor an explicit user request first. This avoids surprising device
    # switches when debugging CPU/MPS numerical or runtime differences.
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        return torch.device("mps")
    if requested == "cuda":
        return torch.device("cuda")

    # For "auto", prefer the fastest local accelerator available in this
    # environment, then fall back to CPU for maximum compatibility.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def subject_id_from_path(path: Path) -> str | None:
    """Extract subject IDs like sub-001 from BIDS-like MRI or mask paths."""

    match = re.search(r"sub-\d+", str(path))
    return match.group(0) if match else None


def find_mri_files(mri_root: Path) -> dict[str, Path]:
    """Find all CAMRI MRI NIfTI volumes and index them by subject ID."""

    # CAMRI images follow a BIDS-like directory layout; the glob intentionally
    # targets the T2w RARE anatomical image and ignores unrelated files.
    pattern = "sub-*/ses-1/anat/sub-*_ses-1_acq-RARE_T2w.nii.gz"
    files = sorted(mri_root.glob(pattern))
    by_subject: dict[str, Path] = {}
    # Store one path per subject ID so downstream matching does not depend on
    # filesystem traversal order.
    for path in files:
        subject_id = subject_id_from_path(path)
        if subject_id is not None:
            by_subject[subject_id] = path
    return by_subject


def find_mask_files(mask_root: Path) -> tuple[dict[str, Path], list[str]]:
    """Find expert masks and index them by subject ID.

    If duplicate masks appear for one subject, the first sorted file is used.
    That choice is deterministic and visible in the skip/warning records.
    """

    # Expert masks are flat in one directory and include an additional suffix,
    # so their glob differs from the MRI glob even though subject IDs match.
    files = sorted(mask_root.glob("CAMRI_Rat-sub-*_ses-1_acq-RARE_T2w_*.nii.gz"))
    by_subject: dict[str, Path] = {}
    warnings: list[str] = []
    for path in files:
        subject_id = subject_id_from_path(path)
        if subject_id is None:
            continue
        if subject_id in by_subject:
            # Duplicate masks would make the benchmark ambiguous. We keep a
            # deterministic first choice and surface the issue in the summary.
            warnings.append(
                f"{subject_id}: multiple masks found; using {by_subject[subject_id].name}"
            )
            continue
        by_subject[subject_id] = path
    return by_subject, warnings


def match_subjects(mri_root: Path, mask_root: Path) -> tuple[list[SubjectPair], list[SkipRecord]]:
    """Match MRI volumes to masks by subject ID and record missing masks.

    Matching by subject ID is more robust than relying on directory order, since
    the CAMRI image and mask folders are organized differently.
    """

    mri_by_subject = find_mri_files(mri_root)
    mask_by_subject, mask_warnings = find_mask_files(mask_root)

    # Convert duplicate-mask warnings into skip-style records so every data
    # quality issue appears in one final report section.
    skips: list[SkipRecord] = []
    for warning in mask_warnings:
        subject_id, reason = warning.split(": ", maxsplit=1)
        skips.append(SkipRecord(subject_id, reason))

    pairs: list[SubjectPair] = []
    # MRI subjects define the benchmark population. A missing mask means we
    # cannot create oracle boxes or compute metrics, so the subject is skipped.
    for subject_id in sorted(mri_by_subject):
        mask_path = mask_by_subject.get(subject_id)
        if mask_path is None:
            skips.append(SkipRecord(subject_id, "missing expert mask"))
            continue
        pairs.append(
            SubjectPair(
                subject_id=subject_id,
                mri_path=mri_by_subject[subject_id],
                mask_path=mask_path,
            )
        )
    return pairs, skips


def load_nifti_data(path: Path) -> np.ndarray:
    """Load a NIfTI file as float32 without changing orientation or spacing."""

    # np.asanyarray(dataobj) loads the array lazily through nibabel while
    # preserving the voxel grid as stored on disk. We do not reorient here
    # because MRI and mask files are already expected to share the same grid.
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def percentile_to_uint8(image_2d: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """MedSAM-style MRI slice normalization.

    This follows the same idea used in the mouse tests and MedSAM's MR
    preprocessing: ignore extreme foreground tails, then scale to 0-255. The
    nonzero foreground guard prevents background from setting the display range
    on sparse slices.
    """

    image_2d = image_2d.astype(np.float32)
    # Exclude NaNs/Infs before percentile estimation; those values would make the
    # display window undefined and break the conversion to uint8.
    finite = image_2d[np.isfinite(image_2d)]
    nonzero = finite[finite > 0]
    # Prefer nonzero foreground voxels because NIfTI backgrounds often contain
    # many zeros that would compress the brain contrast after scaling.
    foreground = nonzero if nonzero.size > 0 else finite
    lower, upper = np.percentile(foreground, [low, high])
    # Clip before scaling so extreme bright voxels do not dominate MedSAM's
    # display-space input.
    clipped = np.clip(image_2d, lower, upper)
    scaled = (clipped - lower) / max(upper - lower, 1e-6)
    return (scaled * 255).astype(np.uint8)


def bbox_from_mask(mask_2d: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    """Create an oracle box from a 2D expert mask.

    Returns:
        [x_min, y_min, x_max, y_max] in image pixel coordinates. The max bounds
        are exclusive, matching NumPy slice semantics and the MedSAM box width
        calculation used later.

    Note:
        NumPy returns row/column coordinates. Rows are y, columns are x.
    """

    # np.where returns y-like row indices first and x-like column indices second.
    rows, cols = np.where(mask_2d > 0)
    if rows.size == 0:
        return None

    height, width = mask_2d.shape
    # Add margin in native pixel space, then clamp to image bounds so every box
    # remains valid even at the brain's edge slices.
    x_min = max(int(cols.min()) - margin, 0)
    x_max = min(int(cols.max()) + margin + 1, width)
    y_min = max(int(rows.min()) - margin, 0)
    y_max = min(int(rows.max()) + margin + 1, height)
    return x_min, y_min, x_max, y_max


def load_medsam_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the MedSAM ViT-B checkpoint in eval mode."""

    # Build the architecture first, then load the downloaded MedSAM weights. This
    # mirrors the local mouse script and avoids invoking the registry checkpoint
    # loader with workspace-specific paths.
    model = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)
    # eval() disables training-time layers and makes inference deterministic for
    # layers such as dropout, even though SAM-like models use little of it.
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def medsam_official_inference(
    medsam_model: torch.nn.Module,
    image_uint8: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    device: torch.device,
) -> np.ndarray:
    """Run MedSAM using the official MedSAM_Inference.py structure.

    MedSAM's released inference script bypasses SamPredictor and calls the
    encoder/decoder modules directly. Following that pathway avoids coordinate
    and preprocessing differences between generic SAM and MedSAM.

    Tensor/coordinate flow:
        - image_uint8: (H, W), display range 0-255
        - image_1024: (1024, 1024, 3), normalized to [0, 1]
        - image_tensor: (1, 3, 1024, 1024)
        - box_torch: (1, 1, 4), scaled from original pixels to 1024 space
        - output: (H, W), binary mask in original slice coordinates
    """

    # Work in the original slice size for outputs, but prepare a 1024-square
    # version for the model because that is the MedSAM checkpoint convention.
    height, width = image_uint8.shape
    image_rgb = np.repeat(image_uint8[:, :, None], 3, axis=-1)

    # The checkpoint expects the same 1024 input geometry used during MedSAM
    # training/inference. preserve_range keeps our 0-255 display scaling intact
    # until the explicit normalization step below.
    image_1024 = transform.resize(
        image_rgb,
        (1024, 1024),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    # The official script normalizes each resized image to [0, 1]. This second
    # normalization is separate from the earlier MRI windowing step.
    image_1024 = (image_1024 - image_1024.min()) / np.clip(
        image_1024.max() - image_1024.min(),
        a_min=1e-8,
        a_max=None,
    )
    # PyTorch expects channel-first tensors. The batch dimension is kept even for
    # one slice because MedSAM modules are batched internally.
    image_tensor = (
        torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # Oracle boxes are generated on the native 256x256 slice. The prompt encoder
    # receives coordinates in the 1024x1024 resized image space.
    box_1024 = np.array([box_xyxy], dtype=np.float32)
    box_1024 = box_1024 / np.array([width, height, width, height]) * 1024
    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=device)
    # MedSAM's prompt encoder expects boxes shaped as (batch, num_boxes, 4).
    box_torch = box_torch[:, None, :]

    # The image embedding is shared by all prompts for a slice. Here there is one
    # oracle box per slice, so we compute it once and immediately decode.
    image_embedding = medsam_model.image_encoder(image_tensor)
    # A box prompt produces sparse embeddings; no point prompts or previous masks
    # are used in this oracle-box benchmark.
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    # mask_decoder combines the image embedding and prompt embedding, producing
    # logits on MedSAM's low-resolution mask grid.
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # The decoder predicts a low-resolution logit mask. Bilinear interpolation
    # brings probabilities back to the original slice grid before thresholding.
    low_res_probs = torch.sigmoid(low_res_logits)
    probs = F.interpolate(
        low_res_probs,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    # Thresholding at 0.5 follows the official MedSAM inference script and gives
    # a binary mask directly comparable to the manual annotation.
    return (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    """Return true positive, false positive, and false negative pixel counts."""

    # Cast once to boolean so all downstream counts are true set operations.
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    # True negatives are not needed for Dice/IoU/precision/recall and would be
    # dominated by background in these mostly empty 2D slices.
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    return tp, fp, fn


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute standard binary segmentation metrics for one 2D slice.

    Empty-mask denominators are handled explicitly so the CSV never contains
    divide-by-zero artifacts. In normal evaluation, ground-truth slices are
    non-empty because empty slices are skipped upstream.
    """

    # All metrics are derived from the same confusion counts so they remain
    # internally consistent for each CSV row.
    tp, fp, fn = confusion_counts(pred, gt)
    pred_area = int(pred.astype(bool).sum())
    gt_area = int(gt.astype(bool).sum())

    # Dice and IoU weight false positives/negatives differently; keeping the
    # denominators explicit makes the metric definitions auditable.
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    # If both masks were empty, overlap metrics are conventionally perfect. That
    # branch rarely triggers here because empty GT slices are skipped upstream.
    dice = 1.0 if dice_den == 0 else (2 * tp) / dice_den
    iou = 1.0 if iou_den == 0 else tp / iou_den
    precision = 1.0 if precision_den == 0 else tp / precision_den
    recall = 1.0 if recall_den == 0 else tp / recall_den

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "mask_area_gt": gt_area,
        "mask_area_pred": pred_area,
        "hausdorff_px": hausdorff_distance_px(pred, gt),
    }


def hausdorff_distance_px(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance in pixels.

    This is optional but useful for boundary outliers. Coordinates are row/col
    pixel indices, so the reported distance is in pixels rather than millimeters.
    If either mask is empty, the value is NaN because the boundary distance is
    undefined.
    """

    # Use all foreground pixels, not just contours. This is simple and stable for
    # thin edge-slice masks where contour extraction can be noisy.
    pred_points = np.column_stack(np.where(pred > 0))
    gt_points = np.column_stack(np.where(gt > 0))
    if pred_points.size == 0 or gt_points.size == 0:
        return float("nan")
    # directed_hausdorff is asymmetric, so compute both directions and retain the
    # larger distance for the standard symmetric Hausdorff value.
    pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
    gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
    return float(max(pred_to_gt, gt_to_pred))


def save_example_figure(
    output_dir: Path,
    subject_id: str,
    slice_index: int,
    image_uint8: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    dice: float,
) -> None:
    """Save a 3-panel visual QC figure for one evaluated slice.

    The prediction panel overlays the MedSAM mask and draws the expert contour,
    which makes boundary disagreements easier to spot than two filled masks.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    x_min, y_min, x_max, y_max = box_xyxy

    # Keep the figure layout fixed across examples so visual QC panels are easy
    # to compare side by side in a file browser.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1 shows the actual prompt given to MedSAM. The cyan rectangle is the
    # oracle box derived from the expert mask.
    axes[0].imshow(image_uint8, cmap="gray")
    axes[0].add_patch(
        plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor="cyan",
            facecolor="none",
            linewidth=1.5,
        )
    )
    axes[0].set_title("MRI + oracle box")

    # Panel 2 shows the manual mask used only for box creation and evaluation.
    axes[1].imshow(image_uint8, cmap="gray")
    axes[1].imshow(gt_mask, alpha=0.35, cmap="Greens")
    axes[1].set_title("Expert mask")

    # Panel 3 overlays MedSAM's prediction and the expert contour, making both
    # boundary offsets and gross failures visible.
    axes[2].imshow(image_uint8, cmap="gray")
    axes[2].imshow(pred_mask, alpha=0.35, cmap="autumn")
    # find_contours returns coordinates as (row, col); Matplotlib expects
    # x=col and y=row, hence the reversed plotting order.
    for contour in measure.find_contours(gt_mask.astype(float), 0.5):
        axes[2].plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1.0)
    axes[2].set_title(f"MedSAM pred | Dice={dice:.3f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    out_path = output_dir / f"{subject_id}_slice-{slice_index:03d}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def csv_fieldnames() -> list[str]:
    """Column order for the per-slice CSV output."""

    # Keeping a fixed order makes downstream plotting scripts and spreadsheet
    # imports reproducible across reruns.
    return [
        "subject_id",
        "slice_index",
        "dice",
        "iou",
        "precision",
        "recall",
        "hausdorff_px",
        "mask_area_gt",
        "mask_area_pred",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]


def evaluate_subject(
    pair: SubjectPair,
    medsam_model: torch.nn.Module,
    device: torch.device,
    box_margin: int,
    example_dir: Path,
    examples_remaining: int,
) -> tuple[list[dict[str, object]], int, str | None]:
    """Evaluate every non-empty expert-mask slice for one subject.

    Returns:
        rows: per-slice metric rows for the CSV.
        examples_saved: number of QC figures saved for this subject.
        skip_reason: reason to skip subject, or None if evaluated.
    """

    # Load both volumes before any per-slice work so shape mismatches are caught
    # once at the subject level.
    mri = load_nifti_data(pair.mri_path)
    gt = load_nifti_data(pair.mask_path) > 0

    if mri.shape != gt.shape:
        return [], 0, f"shape mismatch: MRI {mri.shape} vs mask {gt.shape}"
    if mri.ndim != 3:
        return [], 0, f"expected 3D volume, got shape {mri.shape}"

    # Collect rows in memory for this subject, then append to the global table.
    # This makes it easy to skip a subject cleanly if no slices are usable.
    rows: list[dict[str, object]] = []
    examples_saved = 0

    # CAMRI volumes are stored as (x, y, z); the third axis is the 2D slice index
    # used by the oracle-box benchmark.
    for slice_index in range(mri.shape[2]):
        gt_slice = gt[:, :, slice_index].astype(np.uint8)
        if gt_slice.sum() == 0:
            # Empty ground-truth slices cannot define an oracle box and would
            # inflate easy true-negative metrics, so they are excluded.
            continue

        # The ground-truth mask is used only to create the oracle prompt and to
        # compute metrics after inference. It is never passed to the model as a
        # mask prompt or used to overwrite MedSAM's prediction.
        box = bbox_from_mask(gt_slice, margin=box_margin)
        if box is None:
            continue

        # Normalize each slice independently, matching the 2D MedSAM inference
        # setup and avoiding leakage of intensity statistics across slices.
        image_slice = percentile_to_uint8(mri[:, :, slice_index])
        # This is the only model prediction step. The expert mask has already
        # served its prompt role and is not supplied to MedSAM here.
        pred_slice = medsam_official_inference(medsam_model, image_slice, box, device)
        metrics = segmentation_metrics(pred_slice, gt_slice)

        # Store both metrics and prompt coordinates so poor slices can be traced
        # back to their oracle box without recomputing the benchmark.
        x_min, y_min, x_max, y_max = box
        row = {
            "subject_id": pair.subject_id,
            "slice_index": slice_index,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "hausdorff_px": metrics["hausdorff_px"],
            "mask_area_gt": metrics["mask_area_gt"],
            "mask_area_pred": metrics["mask_area_pred"],
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }
        rows.append(row)

        if examples_saved < examples_remaining:
            # Save a limited number of examples globally, not per subject, to
            # keep the output directory useful rather than overwhelming.
            save_example_figure(
                output_dir=example_dir,
                subject_id=pair.subject_id,
                slice_index=slice_index,
                image_uint8=image_slice,
                gt_mask=gt_slice,
                pred_mask=pred_slice,
                box_xyxy=box,
                dice=float(metrics["dice"]),
            )
            examples_saved += 1

    if not rows:
        return [], examples_saved, "no non-empty expert-mask slices"
    return rows, examples_saved, None


def write_results_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write all per-slice metrics to CSV."""

    # The CSV is intentionally flat: one row per evaluated slice. Per-subject
    # summaries can then be recomputed with different aggregation choices.
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]], evaluated_subjects: set[str], skips: list[SkipRecord]) -> None:
    """Print high-level benchmark statistics."""

    print("\n========== Summary ==========")
    print(f"Subjects evaluated: {len(evaluated_subjects)}")
    print(f"Slices evaluated:   {len(rows)}")

    if rows:
        # Summary statistics here are slice-level to match the CSV granularity.
        # Subject-level summaries are generated by the reporting script.
        dice = np.array([float(row["dice"]) for row in rows], dtype=np.float64)
        iou = np.array([float(row["iou"]) for row in rows], dtype=np.float64)
        print(f"Mean Dice:          {dice.mean():.4f}")
        print(f"Median Dice:        {np.median(dice):.4f}")
        print(f"Mean IoU:           {iou.mean():.4f}")

    print(f"Skipped subjects:   {len(skips)}")
    for skip in skips:
        print(f"  - {skip.subject_id}: {skip.reason}")


def main() -> None:
    """Main benchmark workflow."""

    args = parse_args()
    print("========== CAMRI Rat MedSAM Oracle-Box Evaluation ==========")
    print(f"MRI root:      {args.mri_root}")
    print(f"Mask root:     {args.mask_root}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Results CSV:   {args.results_csv}")
    print(f"Example dir:   {args.example_dir}")
    print(f"Box margin:    {args.box_margin} px")

    # Build the matched subject list before loading the model, so path problems
    # fail fast without spending time on checkpoint initialization.
    pairs, skips = match_subjects(args.mri_root, args.mask_root)
    if args.max_subjects is not None:
        # Truncate after matching so missing-mask accounting remains visible in
        # the final skip list.
        pairs = pairs[: args.max_subjects]
        print(f"Smoke-test mode: evaluating first {len(pairs)} matched subjects")

    print(f"Matched subjects to evaluate: {len(pairs)}")
    if not pairs:
        print("No matched subjects found. Check the folder paths.")
        return

    device = choose_device(args.device)
    print(f"Using device: {device}")
    # The model is loaded once and reused across all slices/subjects.
    medsam_model = load_medsam_model(args.checkpoint, device)

    all_rows: list[dict[str, object]] = []
    evaluated_subjects: set[str] = set()
    examples_remaining = args.max_examples

    for index, pair in enumerate(pairs, start=1):
        # Print paths for traceability; this is helpful when mask suffixes differ
        # across subjects.
        print(f"\n[{index}/{len(pairs)}] {pair.subject_id}")
        print(f"  MRI:  {pair.mri_path}")
        print(f"  Mask: {pair.mask_path}")

        try:
            rows, examples_saved, skip_reason = evaluate_subject(
                pair=pair,
                medsam_model=medsam_model,
                device=device,
                box_margin=args.box_margin,
                example_dir=args.example_dir,
                examples_remaining=examples_remaining,
            )
        except RuntimeError as exc:
            if device.type == "mps":
                # Some PyTorch operations are still less reliable on Apple MPS.
                # If one subject fails there, retry on CPU rather than losing the
                # whole benchmark run.
                print(f"  MPS error: {exc}")
                print("  Falling back to CPU for the rest of the run.")
                device = torch.device("cpu")
                medsam_model.to(device)
                rows, examples_saved, skip_reason = evaluate_subject(
                    pair=pair,
                    medsam_model=medsam_model,
                    device=device,
                    box_margin=args.box_margin,
                    example_dir=args.example_dir,
                    examples_remaining=examples_remaining,
                )
            else:
                raise

        examples_remaining = max(0, examples_remaining - examples_saved)
        if skip_reason is not None:
            print(f"  Skipped: {skip_reason}")
            skips.append(SkipRecord(pair.subject_id, skip_reason))
            continue

        # A subject counts as evaluated only after at least one non-empty slice
        # produced a MedSAM prediction and metrics row.
        all_rows.extend(rows)
        evaluated_subjects.add(pair.subject_id)
        subject_dice = np.array([float(row["dice"]) for row in rows])
        print(
            f"  Evaluated {len(rows)} slices | "
            f"mean Dice={subject_dice.mean():.4f}, "
            f"median Dice={np.median(subject_dice):.4f}"
        )

    write_results_csv(args.results_csv, all_rows)
    print(f"\nSaved per-slice CSV: {args.results_csv}")
    print(f"Saved example figures in: {args.example_dir}")
    print_summary(all_rows, evaluated_subjects, skips)


if __name__ == "__main__":
    main()
