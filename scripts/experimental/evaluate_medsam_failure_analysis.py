"""
Failure analysis for oracle-box MedSAM on CAMRI rat MRI.

Research purpose:
    MedSAM with expert-mask oracle boxes reaches roughly 0.91 Dice on the CAMRI
    rat benchmark, while rodent-specific supervised models can report about
    0.97 Dice. This script is not a replacement benchmark. It is a focused
    diagnostic run that asks where the remaining error lives:

      - Is Dice worse on small masks or edge slices?
      - Are failures concentrated in specific subjects?
      - Do the worst slices show boundary drift, missing tissue, or leakage?

The expert mask is used only to create a 2D oracle box and to evaluate the final
prediction. It is never passed to MedSAM as a mask prompt.

Example full non-empty-slice run:
    MPLBACKEND=Agg ./medsam_env/bin/python scripts/experimental/evaluate_medsam_failure_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Matplotlib otherwise tries to write font/cache files under the user's home
# directory, which may be unavailable in sandboxed research runs.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "medsam_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "medsam_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure


# Reuse the validated CAMRI/MedSAM implementation without editing core code.
REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from evaluate_medsam_camri_rat import (  # noqa: E402
    CHECKPOINT_PATH,
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


DEFAULT_MRI_ROOT = Path("../Datasets/Image_Database/CAMRI Rat Brain MRI Data")
DEFAULT_MASK_ROOT = Path("../Datasets/Mask_Database/RodentBrainMask/CAMRI Rat")
DEFAULT_OUTPUT_DIR = Path("outputs/failure_analysis")
DEFAULT_BOX_MARGIN = 5
DEFAULT_MAX_SUBJECTS = 5
DEFAULT_MAX_SLICES_PER_SUBJECT = 0
EASY_SUBJECT_IDS = [f"sub-{idx:03d}" for idx in range(1, 6)]


@dataclass
class FailureSlice:
    """One selected non-empty 2D slice plus its subject-local position."""

    subject_id: str
    slice_index: int
    slice_position: int
    slice_fraction: float
    image_2d: np.ndarray
    gt_mask: np.ndarray


@dataclass
class SelectionSummary:
    """Record whether the run used every non-empty slice or a sampled subset."""

    selected_subject_ids: list[str]
    subjects_loaded: int
    subjects_with_slices: int
    total_non_empty_slices: int
    selected_slices: int
    max_slices_per_subject: int

    @property
    def used_all_non_empty_slices(self) -> bool:
        """True when no per-subject slice sampling was requested."""

        return self.max_slices_per_subject <= 0


@dataclass
class EvaluatedSlice:
    """All information needed for metrics, plots, and later QC selection."""

    subject_id: str
    slice_index: int
    slice_position: int
    slice_fraction: float
    image_uint8: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    box_xyxy: tuple[int, int, int, int]
    metrics: dict[str, float]


@dataclass
class OutputPaths:
    """Structured output locations for one failure-analysis run."""

    base_dir: Path
    run_dir: Path
    tables_dir: Path
    figures_dir: Path
    overlays_dir: Path
    latest_dir: Path
    comparison_dir: Path


def parse_args() -> argparse.Namespace:
    """Read options for a small, reproducible failure-analysis run."""

    parser = argparse.ArgumentParser(
        description="Failure analysis for oracle-box MedSAM on CAMRI rat MRI."
    )
    parser.add_argument("--mri-root", type=Path, default=DEFAULT_MRI_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional run folder name. When provided, outputs are saved under "
            "outputs/failure_analysis/runs/<run-name>/ and copied to latest/."
        ),
    )
    parser.add_argument(
        "--compare-runs",
        nargs=2,
        metavar=("EASY_RUN", "HARD_RUN"),
        default=None,
        help=(
            "Compare two completed runs from outputs/failure_analysis/runs/. "
            "This mode writes only the comparison summary and does not run MedSAM."
        ),
    )
    parser.add_argument("--box-margin", type=int, default=DEFAULT_BOX_MARGIN)
    parser.add_argument("--max-subjects", type=int, default=DEFAULT_MAX_SUBJECTS)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help=(
            "Specific subject IDs to evaluate, for example: "
            "--subjects sub-001 sub-045 sub-088. Overrides --max-subjects."
        ),
    )
    parser.add_argument(
        "--max-slices-per-subject",
        type=int,
        default=DEFAULT_MAX_SLICES_PER_SUBJECT,
        help=(
            "Evaluate all non-empty slices when 0. If positive, uniformly sample "
            "up to this many non-empty mask slices per subject."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto uses MPS/CUDA if available, otherwise CPU.",
    )
    return parser.parse_args()


def choose_non_empty_slices(mask_volume: np.ndarray, max_slices: int) -> list[int]:
    """Return all non-empty slices, or uniformly sample them when requested."""

    non_empty = [idx for idx in range(mask_volume.shape[2]) if mask_volume[:, :, idx].sum() > 0]
    if max_slices <= 0:
        return non_empty
    if len(non_empty) <= max_slices:
        return non_empty

    # Uniform spacing makes this a compact failure-analysis sample rather than a
    # center-slice-only snapshot. Rounding may collide, so remove duplicates.
    positions = np.linspace(0, len(non_empty) - 1, max_slices)
    return sorted({non_empty[int(round(pos))] for pos in positions})


def select_subject_pairs(
    mri_root: Path,
    mask_root: Path,
    max_subjects: int,
    requested_subjects: list[str] | None,
) -> tuple[list[SubjectPair], list[str]]:
    """Select matched subjects by explicit IDs or by the default first-N rule."""

    pairs, skips = match_subjects(mri_root, mask_root)
    if skips:
        print(f"Data warnings before truncation: {len(skips)}")
        for skip in skips[:8]:
            print(f"  - {skip.subject_id}: {skip.reason}")

    if requested_subjects is None:
        selected_pairs = pairs[:max_subjects]
        return selected_pairs, [pair.subject_id for pair in selected_pairs]

    pair_by_subject = {pair.subject_id: pair for pair in pairs}
    selected_pairs: list[SubjectPair] = []
    selected_subject_ids: list[str] = []
    missing_subjects: list[str] = []
    seen_subjects: set[str] = set()

    for subject_id in requested_subjects:
        if subject_id in seen_subjects:
            continue
        seen_subjects.add(subject_id)

        pair = pair_by_subject.get(subject_id)
        if pair is None:
            missing_subjects.append(subject_id)
            continue
        selected_pairs.append(pair)
        selected_subject_ids.append(subject_id)

    if missing_subjects:
        print(
            "Warning: requested subjects not found in matched MRI/mask pairs: "
            + ", ".join(missing_subjects)
        )

    return selected_pairs, selected_subject_ids


def resolve_output_paths(base_dir: Path, run_name: str | None) -> OutputPaths:
    """Create the structured output paths for this run."""

    run_dir = base_dir / "runs" / run_name if run_name else base_dir / "latest"
    return OutputPaths(
        base_dir=base_dir,
        run_dir=run_dir,
        tables_dir=run_dir / "tables",
        figures_dir=run_dir / "figures",
        overlays_dir=run_dir / "overlays",
        latest_dir=base_dir / "latest",
        comparison_dir=base_dir / "comparison",
    )


def prepare_run_directory(paths: OutputPaths) -> None:
    """Start a run output directory from a clean state."""

    if paths.run_dir.exists():
        shutil.rmtree(paths.run_dir)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.overlays_dir.mkdir(parents=True, exist_ok=True)


def update_latest_run(paths: OutputPaths) -> None:
    """Mirror the most recent named run into outputs/failure_analysis/latest."""

    if paths.run_dir == paths.latest_dir:
        return
    if paths.latest_dir.exists():
        shutil.rmtree(paths.latest_dir)
    shutil.copytree(paths.run_dir, paths.latest_dir)


def load_failure_slices(
    mri_root: Path,
    mask_root: Path,
    max_subjects: int,
    max_slices_per_subject: int,
    requested_subjects: list[str] | None,
) -> tuple[list[FailureSlice], SelectionSummary]:
    """Load selected slices from matched CAMRI MRI/mask pairs."""

    selected_pairs, selected_subject_ids = select_subject_pairs(
        mri_root=mri_root,
        mask_root=mask_root,
        max_subjects=max_subjects,
        requested_subjects=requested_subjects,
    )

    records: list[FailureSlice] = []
    total_non_empty_slices = 0
    subjects_with_slices = 0
    for pair_index, pair in enumerate(selected_pairs, start=1):
        print(f"[{pair_index}/{len(selected_pairs)}] Loading {pair.subject_id}")
        mri = load_nifti_data(pair.mri_path)
        gt = load_nifti_data(pair.mask_path) > 0

        if mri.shape != gt.shape:
            print(f"  Skipped shape mismatch: MRI {mri.shape} vs mask {gt.shape}")
            continue
        if mri.ndim != 3:
            print(f"  Skipped non-3D volume: {mri.shape}")
            continue

        non_empty_count = sum(
            1 for idx in range(gt.shape[2]) if gt[:, :, idx].sum() > 0
        )
        total_non_empty_slices += non_empty_count
        selected_indices = choose_non_empty_slices(gt, max_slices_per_subject)
        if selected_indices:
            subjects_with_slices += 1
        denominator = max(len(selected_indices) - 1, 1)
        if max_slices_per_subject <= 0:
            print(f"  Selected all {len(selected_indices)} non-empty slices")
        else:
            print(
                f"  Selected {len(selected_indices)} of "
                f"{non_empty_count} non-empty slices"
            )

        for position, slice_index in enumerate(selected_indices):
            records.append(
                FailureSlice(
                    subject_id=pair.subject_id,
                    slice_index=slice_index,
                    slice_position=position,
                    slice_fraction=position / denominator,
                    image_2d=mri[:, :, slice_index].astype(np.float32),
                    gt_mask=gt[:, :, slice_index].astype(np.uint8),
                )
            )

    summary = SelectionSummary(
        selected_subject_ids=selected_subject_ids,
        subjects_loaded=len(selected_pairs),
        subjects_with_slices=subjects_with_slices,
        total_non_empty_slices=total_non_empty_slices,
        selected_slices=len(records),
        max_slices_per_subject=max_slices_per_subject,
    )
    return records, summary


def evaluate_slices(
    records: list[FailureSlice],
    medsam_model: torch.nn.Module,
    device: torch.device,
    box_margin: int,
) -> list[EvaluatedSlice]:
    """Run MedSAM once per selected slice using expert-mask oracle boxes."""

    evaluated: list[EvaluatedSlice] = []
    for record_index, record in enumerate(records, start=1):
        print(
            f"[{record_index}/{len(records)}] Inference "
            f"{record.subject_id} slice {record.slice_index}"
        )
        box = bbox_from_mask(record.gt_mask, margin=box_margin)
        if box is None:
            continue

        image_uint8 = percentile_to_uint8(record.image_2d)
        pred_mask = medsam_official_inference(medsam_model, image_uint8, box, device)
        metrics = segmentation_metrics(pred_mask, record.gt_mask)

        # The requested mask_area and box_area are ground-truth quantities. The
        # prediction area is included too because FP leakage can explain Dice
        # loss even when the oracle prompt is valid.
        x_min, y_min, x_max, y_max = box
        metrics["mask_area"] = float(record.gt_mask.astype(bool).sum())
        metrics["pred_area"] = float(pred_mask.astype(bool).sum())
        metrics["box_area"] = float((x_max - x_min) * (y_max - y_min))

        evaluated.append(
            EvaluatedSlice(
                subject_id=record.subject_id,
                slice_index=record.slice_index,
                slice_position=record.slice_position,
                slice_fraction=record.slice_fraction,
                image_uint8=image_uint8,
                gt_mask=record.gt_mask,
                pred_mask=pred_mask,
                box_xyxy=box,
                metrics=metrics,
            )
        )

    return evaluated


def finite_mean(values: Iterable[float]) -> float:
    """Return a NaN-aware mean for CSV summaries."""

    array = np.array(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def finite_std(values: Iterable[float]) -> float:
    """Return a NaN-aware sample standard deviation for CSV summaries."""

    array = np.array(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size <= 1:
        return float("nan")
    return float(finite.std(ddof=1))


def per_slice_row(item: EvaluatedSlice) -> dict[str, object]:
    """Convert one evaluated slice into the requested per-slice CSV schema."""

    x_min, y_min, x_max, y_max = item.box_xyxy
    return {
        "subject": item.subject_id,
        "slice_index": item.slice_index,
        "slice_position": item.slice_position,
        "slice_fraction": item.slice_fraction,
        "dice": item.metrics["dice"],
        "iou": item.metrics["iou"],
        "precision": item.metrics["precision"],
        "recall": item.metrics["recall"],
        "hausdorff": item.metrics["hausdorff_px"],
        "mask_area": item.metrics["mask_area"],
        "pred_area": item.metrics["pred_area"],
        "box_area": item.metrics["box_area"],
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def write_per_slice_metrics(path: Path, evaluated: list[EvaluatedSlice]) -> None:
    """Write the detailed per-slice failure-analysis table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "slice_index",
        "slice_position",
        "slice_fraction",
        "dice",
        "iou",
        "precision",
        "recall",
        "hausdorff",
        "mask_area",
        "pred_area",
        "box_area",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in evaluated:
            writer.writerow(per_slice_row(item))


def summarize_group(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    """Summarize metrics by one categorical column."""

    grouped: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        dice = [float(row["dice"]) for row in group]
        iou = [float(row["iou"]) for row in group]
        precision = [float(row["precision"]) for row in group]
        recall = [float(row["recall"]) for row in group]
        hausdorff = [float(row["hausdorff"]) for row in group]
        mask_area = [float(row["mask_area"]) for row in group]
        box_area = [float(row["box_area"]) for row in group]

        summary_rows.append(
            {
                group_key: key,
                "n_slices": len(group),
                "mean_dice": finite_mean(dice),
                "std_dice": finite_std(dice),
                "min_dice": float(np.min(dice)),
                "median_dice": float(np.median(dice)),
                "max_dice": float(np.max(dice)),
                "mean_iou": finite_mean(iou),
                "mean_precision": finite_mean(precision),
                "mean_recall": finite_mean(recall),
                "mean_hausdorff": finite_mean(hausdorff),
                "mean_mask_area": finite_mean(mask_area),
                "mean_box_area": finite_mean(box_area),
            }
        )
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a summary table with stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows_markdown(path: Path, rows: list[dict[str, object]], title: str) -> None:
    """Write a readable Markdown table for a list of dictionaries."""

    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available.")
        path.write_text("\n".join(lines) + "\n")
        return

    fieldnames = list(rows[0].keys())
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("|" + "|".join(["---"] * len(fieldnames)) + "|")
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV table when it exists, otherwise return an empty table."""

    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def float_value(row: dict[str, object], key: str) -> float:
    """Convert a CSV value to float while keeping missing values as NaN."""

    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def metric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    """Summarize the presentation metrics for a group of per-slice rows."""

    dice = [float_value(row, "dice") for row in rows]
    precision = [float_value(row, "precision") for row in rows]
    recall = [float_value(row, "recall") for row in rows]
    mask_area = [float_value(row, "mask_area") for row in rows]
    box_area = [float_value(row, "box_area") for row in rows]

    return {
        "n_slices": float(len(rows)),
        "mean_dice": finite_mean(dice),
        "median_dice": float(np.nanmedian(dice)) if rows else float("nan"),
        "worst_dice": float(np.nanmin(dice)) if rows else float("nan"),
        "best_dice": float(np.nanmax(dice)) if rows else float("nan"),
        "mean_precision": finite_mean(precision),
        "mean_recall": finite_mean(recall),
        "mean_mask_area": finite_mean(mask_area),
        "mean_box_area": finite_mean(box_area),
    }


def format_float(value: float, digits: int = 4) -> str:
    """Format values for presentation tables without exposing NaN internals."""

    if not np.isfinite(value):
        return "not available"
    return f"{value:.{digits}f}"


def group_rows_by_subject(
    rows: list[dict[str, object]],
    subject_ids: list[str],
) -> list[dict[str, object]]:
    """Return rows whose subject is in the requested ID list."""

    subject_set = set(subject_ids)
    return [row for row in rows if str(row.get("subject", "")) in subject_set]


def count_subjects(rows: list[dict[str, object]]) -> int:
    """Count unique subjects represented in a per-slice table."""

    return len({str(row.get("subject", "")) for row in rows if row.get("subject")})


def infer_failure_type(row: dict[str, object]) -> str:
    """Use precision/recall balance to name the likely failure mode."""

    precision = float_value(row, "precision")
    recall = float_value(row, "recall")
    low_threshold = 0.85
    difference_threshold = 0.02

    if precision < low_threshold and recall < low_threshold:
        return "complete failure"
    if precision + difference_threshold < recall:
        return "over-segmentation / wrong structure"
    if recall + difference_threshold < precision:
        return "under-segmentation / missed brain"
    return "mixed boundary error"


def worst_slice_label(row: dict[str, object]) -> str:
    """Create a compact subject/slice/Dice label for the worst failure."""

    return (
        f"{row.get('subject', 'unknown')}, slice {row.get('slice_index', 'unknown')}, "
        f"Dice {format_float(float_value(row, 'dice'))}"
    )


def build_presentation_summary_rows(
    per_slice_rows: list[dict[str, object]],
    summary_by_subject_rows: list[dict[str, object]],
    selected_subject_ids: list[str],
) -> list[dict[str, str]]:
    """Build one concise CSV-friendly summary for slides and reports."""

    overall = metric_summary(per_slice_rows)
    worst_row = min(per_slice_rows, key=lambda row: float_value(row, "dice"))
    worst_10 = sorted(per_slice_rows, key=lambda row: float_value(row, "dice"))[:10]
    worst_10_area = finite_mean(float_value(row, "mask_area") for row in worst_10)
    all_area = finite_mean(float_value(row, "mask_area") for row in per_slice_rows)
    mean_minus_median = overall["mean_dice"] - overall["median_dice"]
    subject_ids = [
        str(row.get("subject", ""))
        for row in summary_by_subject_rows
        if row.get("subject")
    ]

    return [
        {
            "section": "overall_run_summary",
            "item": "selected_subjects",
            "value": ", ".join(selected_subject_ids or subject_ids),
        },
        {
            "section": "overall_run_summary",
            "item": "number_of_subjects",
            "value": str(count_subjects(per_slice_rows)),
        },
        {
            "section": "overall_run_summary",
            "item": "number_of_slices",
            "value": str(len(per_slice_rows)),
        },
        {
            "section": "overall_run_summary",
            "item": "mean_dice",
            "value": format_float(overall["mean_dice"]),
        },
        {
            "section": "overall_run_summary",
            "item": "median_dice",
            "value": format_float(overall["median_dice"]),
        },
        {
            "section": "overall_run_summary",
            "item": "worst_dice",
            "value": format_float(overall["worst_dice"]),
        },
        {
            "section": "overall_run_summary",
            "item": "best_dice",
            "value": format_float(overall["best_dice"]),
        },
        {
            "section": "failure_interpretation",
            "item": "typical_performance",
            "value": f"median Dice {format_float(overall['median_dice'])}",
        },
        {
            "section": "failure_interpretation",
            "item": "outlier_effect",
            "value": (
                "mean Dice is lower than median Dice "
                f"by {format_float(abs(mean_minus_median))}"
                if mean_minus_median < 0
                else "mean Dice is not lower than median Dice"
            ),
        },
        {
            "section": "failure_interpretation",
            "item": "worst_failure",
            "value": worst_slice_label(worst_row),
        },
        {
            "section": "failure_interpretation",
            "item": "main_suspected_failure_type",
            "value": infer_failure_type(worst_row),
        },
        {
            "section": "failure_interpretation",
            "item": "small_slice_evidence",
            "value": (
                f"worst 10 mean mask area {format_float(worst_10_area, 1)} px; "
                f"all-slice mean {format_float(all_area, 1)} px"
            ),
        },
    ]


def write_presentation_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the presentation summary as a compact long-form CSV."""

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "item", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(lines: list[str], rows: list[tuple[str, str]]) -> None:
    """Append a two-column Markdown table to an existing list of lines."""

    lines.append("| Item | Value |")
    lines.append("|---|---|")
    for item, value in rows:
        lines.append(f"| {item} | {value} |")
    lines.append("")


def write_presentation_summary_md(
    path: Path,
    summary_rows: list[dict[str, str]],
) -> None:
    """Write a concise Markdown summary suitable for presentation notes."""

    by_section: dict[str, list[dict[str, str]]] = {}
    for row in summary_rows:
        by_section.setdefault(row["section"], []).append(row)

    lines = ["# MedSAM Failure Analysis Summary", ""]
    lines.append("## Overall Run Summary")
    write_markdown_table(
        lines,
        [(row["item"].replace("_", " ").title(), row["value"]) for row in by_section["overall_run_summary"]],
    )

    lines.append("## Failure Interpretation")
    write_markdown_table(
        lines,
        [(row["item"].replace("_", " ").title(), row["value"]) for row in by_section["failure_interpretation"]],
    )

    path.write_text("\n".join(lines) + "\n")


def build_comparison_rows(
    easy_rows: list[dict[str, object]],
    hard_rows: list[dict[str, object]],
    easy_name: str,
    hard_name: str,
) -> list[dict[str, object]]:
    """Build Easy vs Hard comparison rows from two completed runs."""

    comparison_rows: list[dict[str, object]] = []
    for group_name, rows in [(easy_name, easy_rows), (hard_name, hard_rows)]:
        summary = metric_summary(rows)
        comparison_rows.append(
            {
                "group": group_name,
                "subjects": ", ".join(sorted({str(row.get("subject", "")) for row in rows})),
                "n_subjects": count_subjects(rows),
                "n_slices": int(summary["n_slices"]),
                "mean_dice": summary["mean_dice"],
                "median_dice": summary["median_dice"],
                "worst_dice": summary["worst_dice"],
                "best_dice": summary["best_dice"],
                "mean_precision": summary["mean_precision"],
                "mean_recall": summary["mean_recall"],
                "mean_mask_area": summary["mean_mask_area"],
                "mean_box_area": summary["mean_box_area"],
            }
        )
    return comparison_rows


def write_easy_vs_hard_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the Easy vs Hard comparison CSV."""

    fieldnames = [
        "group",
        "subjects",
        "n_subjects",
        "n_slices",
        "mean_dice",
        "median_dice",
        "worst_dice",
        "best_dice",
        "mean_precision",
        "mean_recall",
        "mean_mask_area",
        "mean_box_area",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_easy_vs_hard_md(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the Easy vs Hard comparison as a readable Markdown table."""

    lines = ["# Easy vs Hard Failure-Analysis Comparison", ""]
    lines.append(
        "| Run | Subjects | N Subjects | N Slices | Mean Dice | Median Dice | "
        "Worst Dice | Best Dice | Precision | Recall | Mask Area | Box Area |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {group} | {subjects} | {n_subjects} | {n_slices} | {mean_dice} | "
            "{median_dice} | {worst_dice} | {best_dice} | {precision} | "
            "{recall} | {mask_area} | {box_area} |".format(
                group=row["group"],
                subjects=row["subjects"],
                n_subjects=row["n_subjects"],
                n_slices=row["n_slices"],
                mean_dice=format_float(float(row["mean_dice"])),
                median_dice=format_float(float(row["median_dice"])),
                worst_dice=format_float(float(row["worst_dice"])),
                best_dice=format_float(float(row["best_dice"])),
                precision=format_float(float(row["mean_precision"])),
                recall=format_float(float(row["mean_recall"])),
                mask_area=format_float(float(row["mean_mask_area"]), 1),
                box_area=format_float(float(row["mean_box_area"]), 1),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def comparison_value(row: dict[str, object], key: str, digits: int) -> str:
    """Format one comparison value for the simple presentation table."""

    value = float(row[key])
    if not np.isfinite(value):
        return "not available"
    return f"{value:.{digits}f}"


def comparison_area(row: dict[str, object], key: str) -> str:
    """Format area values as whole pixels for presentation."""

    value = float(row[key])
    if not np.isfinite(value):
        return "not available"
    return f"{value:.0f}"


def build_simple_presentation_comparison(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Build the exact Metric/Easy/Hard table requested for slides."""

    easy_row, hard_row = rows
    return [
        {
            "Metric": "Mean Dice",
            "Easy": comparison_value(easy_row, "mean_dice", 4),
            "Hard": comparison_value(hard_row, "mean_dice", 4),
        },
        {
            "Metric": "Median Dice",
            "Easy": comparison_value(easy_row, "median_dice", 4),
            "Hard": comparison_value(hard_row, "median_dice", 4),
        },
        {
            "Metric": "Worst Dice",
            "Easy": comparison_value(easy_row, "worst_dice", 4),
            "Hard": comparison_value(hard_row, "worst_dice", 4),
        },
        {
            "Metric": "Best Dice",
            "Easy": comparison_value(easy_row, "best_dice", 4),
            "Hard": comparison_value(hard_row, "best_dice", 4),
        },
        {
            "Metric": "Mean Precision",
            "Easy": comparison_value(easy_row, "mean_precision", 4),
            "Hard": comparison_value(hard_row, "mean_precision", 4),
        },
        {
            "Metric": "Mean Recall",
            "Easy": comparison_value(easy_row, "mean_recall", 4),
            "Hard": comparison_value(hard_row, "mean_recall", 4),
        },
        {
            "Metric": "Mean Brain Area",
            "Easy": comparison_area(easy_row, "mean_mask_area"),
            "Hard": comparison_area(hard_row, "mean_mask_area"),
        },
        {
            "Metric": "Mean Box Area",
            "Easy": comparison_area(easy_row, "mean_box_area"),
            "Hard": comparison_area(hard_row, "mean_box_area"),
        },
        {
            "Metric": "Subjects",
            "Easy": str(easy_row["n_subjects"]),
            "Hard": str(hard_row["n_subjects"]),
        },
        {
            "Metric": "Slices",
            "Easy": str(easy_row["n_slices"]),
            "Hard": str(hard_row["n_slices"]),
        },
    ]


def write_simple_presentation_comparison_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write the slide-friendly Metric/Easy/Hard comparison CSV."""

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Easy", "Hard"])
        writer.writeheader()
        writer.writerows(rows)


def write_simple_presentation_comparison_md(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write the slide-friendly Metric/Easy/Hard comparison Markdown table."""

    lines = ["| Metric | Easy | Hard |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['Metric']} | {row['Easy']} | {row['Hard']} |")
    path.write_text("\n".join(lines) + "\n")


def save_easy_vs_hard_plot(path: Path, rows: list[dict[str, object]]) -> None:
    """Save one clean bar plot comparing core metrics for Easy and Hard groups."""

    metrics = [
        ("mean_dice", "Mean Dice"),
        ("median_dice", "Median Dice"),
        ("mean_precision", "Precision"),
        ("mean_recall", "Recall"),
    ]
    groups = [str(row["group"]) for row in rows]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))
    for group_index, row in enumerate(rows):
        values = [float(row[key]) if np.isfinite(float(row[key])) else 0.0 for key, _ in metrics]
        offset = (group_index - (len(rows) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=groups[group_index])

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Easy vs Hard MedSAM Performance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_compare_mode(base_dir: Path, easy_run: str, hard_run: str) -> None:
    """Compare two completed run folders and write presentation-ready outputs."""

    easy_csv = base_dir / "runs" / easy_run / "tables" / "per_slice_metrics.csv"
    hard_csv = base_dir / "runs" / hard_run / "tables" / "per_slice_metrics.csv"
    easy_rows = read_csv_rows(easy_csv)
    hard_rows = read_csv_rows(hard_csv)

    if not easy_rows:
        print(f"Missing or empty easy run table: {easy_csv}")
        return
    if not hard_rows:
        print(f"Missing or empty hard run table: {hard_csv}")
        return

    comparison_dir = base_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = build_comparison_rows(
        easy_rows=easy_rows,
        hard_rows=hard_rows,
        easy_name=easy_run,
        hard_name=hard_run,
    )
    simple_rows = build_simple_presentation_comparison(comparison_rows)

    write_easy_vs_hard_csv(
        comparison_dir / "easy_vs_hard_summary.csv",
        comparison_rows,
    )
    write_easy_vs_hard_md(
        comparison_dir / "easy_vs_hard_summary.md",
        comparison_rows,
    )
    save_easy_vs_hard_plot(
        comparison_dir / "easy_vs_hard_summary.png",
        comparison_rows,
    )
    write_simple_presentation_comparison_csv(
        comparison_dir / "presentation_easy_vs_hard_table.csv",
        simple_rows,
    )
    write_simple_presentation_comparison_md(
        comparison_dir / "presentation_easy_vs_hard_table.md",
        simple_rows,
    )

    print("========== Easy vs Hard Comparison Complete ==========")
    print(f"Easy run:   {easy_run}")
    print(f"Hard run:   {hard_run}")
    print(f"Outputs:    {comparison_dir}")


def save_presentation_outputs(
    tables_dir: Path,
    selection_summary: SelectionSummary,
) -> None:
    """Create concise presentation-ready summaries from the saved CSV outputs."""

    per_slice_rows = read_csv_rows(tables_dir / "per_slice_metrics.csv")
    summary_by_subject_rows = read_csv_rows(tables_dir / "summary_by_subject.csv")
    if not per_slice_rows or not summary_by_subject_rows:
        print("Presentation summary skipped: expected metric CSVs were not found.")
        return

    summary_rows = build_presentation_summary_rows(
        per_slice_rows=per_slice_rows,
        summary_by_subject_rows=summary_by_subject_rows,
        selected_subject_ids=selection_summary.selected_subject_ids,
    )

    write_presentation_summary_csv(tables_dir / "presentation_summary.csv", summary_rows)
    write_presentation_summary_md(
        tables_dir / "presentation_summary.md",
        summary_rows,
    )


def save_dice_by_slice_index(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot mean Dice by sampled subject-local slice position with std bands."""

    summary = summarize_group(rows, "slice_position")
    positions = np.array([float(row["slice_position"]) for row in summary])
    means = np.array([float(row["mean_dice"]) for row in summary])
    stds = np.array([float(row["std_dice"]) for row in summary])
    stds = np.nan_to_num(stds, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positions, means, marker="o", linewidth=2.0, color="#1f77b4")
    ax.fill_between(positions, means - stds, means + stds, color="#1f77b4", alpha=0.2)
    ax.set_xlabel("Sampled non-empty slice position")
    ax.set_ylabel("Dice")
    ax.set_title("MedSAM Dice by Slice Position")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_dice_distribution(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot the distribution of per-slice Dice scores."""

    dice = [float(row["dice"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dice, bins=np.linspace(0.0, 1.0, 21), color="#4c78a8", edgecolor="white")
    ax.axvline(finite_mean(dice), color="#d62728", linewidth=1.8, label="mean")
    ax.set_xlabel("Dice")
    ax.set_ylabel("Slice count")
    ax.set_title("Per-Slice Dice Distribution")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_dice_vs_mask_area(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot Dice against expert-mask area to expose small-slice sensitivity."""

    mask_area = np.array([float(row["mask_area"]) for row in rows])
    dice = np.array([float(row["dice"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mask_area, dice, s=34, alpha=0.75, color="#2ca02c", edgecolor="none")
    ax.set_xlabel("Expert mask area (pixels)")
    ax.set_ylabel("Dice")
    ax.set_title("Dice vs Mask Area")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_boundary(ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float) -> None:
    """Draw a binary mask boundary on an existing image axis."""

    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def save_overlay(path: Path, item: EvaluatedSlice, label: str) -> None:
    """Save one QC overlay with MRI, GT boundary, MedSAM boundary, and metadata."""

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(item.image_uint8, cmap="gray")
    draw_boundary(ax, item.gt_mask, color="red", linewidth=1.4)
    draw_boundary(ax, item.pred_mask, color="lime", linewidth=1.4)
    ax.set_title(
        f"{label} | Dice={item.metrics['dice']:.3f}\n"
        f"{item.subject_id} | slice {item.slice_index}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_qc_examples(evaluated: list[EvaluatedSlice]) -> dict[str, list[EvaluatedSlice]]:
    """Select best, median-neighborhood, and worst slices by Dice."""

    ranked = sorted(evaluated, key=lambda item: float(item.metrics["dice"]))
    if not ranked:
        return {"best_5": [], "median_5": [], "worst_10": []}

    median_index = len(ranked) // 2
    median_start = max(0, median_index - 2)
    median_end = min(len(ranked), median_start + 5)
    median_start = max(0, median_end - 5)

    return {
        "best_5": list(reversed(ranked[-5:])),
        "median_5": ranked[median_start:median_end],
        "worst_10": ranked[:10],
    }


def save_qc_overlays(output_dir: Path, evaluated: list[EvaluatedSlice]) -> None:
    """Save ranked QC overlays into best_5, median_5, and worst_10 folders."""

    for folder_name, items in choose_qc_examples(evaluated).items():
        folder = output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(items, start=1):
            out_path = folder / (
                f"{rank:02d}_{item.subject_id}_slice-{item.slice_index:03d}_"
                f"dice-{item.metrics['dice']:.3f}.png"
            )
            save_overlay(out_path, item, folder_name)


def print_run_summary(
    evaluated: list[EvaluatedSlice],
    output_dir: Path,
    selection_summary: SelectionSummary,
) -> None:
    """Print one clear terminal summary for interpreting the run."""

    dice = [float(item.metrics["dice"]) for item in evaluated]
    if selection_summary.used_all_non_empty_slices:
        slice_mode = "all non-empty slices were used"
    else:
        slice_mode = (
            f"uniformly sampled up to "
            f"{selection_summary.max_slices_per_subject} non-empty slices per subject"
        )

    print("\n========== Failure Analysis Complete ==========")
    print(
        "Selected subjects:      "
        + (
            ", ".join(selection_summary.selected_subject_ids)
            if selection_summary.selected_subject_ids
            else "none"
        )
    )
    print(f"Subjects loaded:        {selection_summary.subjects_loaded}")
    print(f"Subjects evaluated:     {selection_summary.subjects_with_slices}")
    print(f"Slice selection:        {slice_mode}")
    print(f"Non-empty slices found: {selection_summary.total_non_empty_slices}")
    print(f"Slices evaluated:       {len(evaluated)}")
    if dice:
        print(f"Mean Dice:              {finite_mean(dice):.4f}")
        print(f"Median Dice:            {float(np.median(dice)):.4f}")
        print(f"Worst Dice:             {float(np.min(dice)):.4f}")
        print(f"Best Dice:              {float(np.max(dice)):.4f}")
    print(f"Outputs:                {output_dir}")


def main() -> None:
    """Run the CAMRI rat MedSAM failure-analysis experiment."""

    args = parse_args()
    base_output_dir = args.output_dir
    if args.compare_runs:
        easy_run, hard_run = args.compare_runs
        run_compare_mode(base_output_dir, easy_run, hard_run)
        return

    paths = resolve_output_paths(base_output_dir, args.run_name)
    prepare_run_directory(paths)

    print("========== MedSAM CAMRI Rat Failure Analysis ==========")
    print(f"MRI root:            {args.mri_root}")
    print(f"Mask root:           {args.mask_root}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Output dir:          {paths.run_dir}")
    if args.run_name:
        print(f"Run name:            {args.run_name}")
    print(f"Box margin:          {args.box_margin} px")
    if args.subjects:
        print(f"Subjects:            {', '.join(args.subjects)}")
    else:
        print(f"Max subjects:        {args.max_subjects}")
    print(f"Max slices/subject:  {args.max_slices_per_subject}")

    records, selection_summary = load_failure_slices(
        mri_root=args.mri_root,
        mask_root=args.mask_root,
        max_subjects=args.max_subjects,
        max_slices_per_subject=args.max_slices_per_subject,
        requested_subjects=args.subjects,
    )
    if not records:
        print("No evaluable slices found. Check dataset paths and masks.")
        return

    device = choose_device(args.device)
    print(f"Using device:        {device}")
    medsam_model = load_medsam_model(args.checkpoint, device)

    evaluated = evaluate_slices(
        records=records,
        medsam_model=medsam_model,
        device=device,
        box_margin=args.box_margin,
    )
    rows = [per_slice_row(item) for item in evaluated]
    summary_by_subject = summarize_group(rows, "subject")
    summary_by_slice_position = summarize_group(rows, "slice_position")

    write_per_slice_metrics(paths.tables_dir / "per_slice_metrics.csv", evaluated)
    write_summary_csv(
        paths.tables_dir / "summary_by_subject.csv",
        summary_by_subject,
    )
    write_rows_markdown(
        paths.tables_dir / "summary_by_subject.md",
        summary_by_subject,
        "Summary By Subject",
    )
    write_summary_csv(
        paths.tables_dir / "summary_by_slice_position.csv",
        summary_by_slice_position,
    )
    write_rows_markdown(
        paths.tables_dir / "summary_by_slice_position.md",
        summary_by_slice_position,
        "Summary By Slice Position",
    )

    save_dice_by_slice_index(paths.figures_dir / "dice_by_slice_index.png", rows)
    save_dice_distribution(paths.figures_dir / "dice_distribution_histogram.png", rows)
    save_dice_vs_mask_area(paths.figures_dir / "dice_vs_mask_area.png", rows)
    save_qc_overlays(paths.overlays_dir, evaluated)
    save_presentation_outputs(
        tables_dir=paths.tables_dir,
        selection_summary=selection_summary,
    )

    update_latest_run(paths)
    print_run_summary(evaluated, paths.run_dir, selection_summary)


if __name__ == "__main__":
    main()
