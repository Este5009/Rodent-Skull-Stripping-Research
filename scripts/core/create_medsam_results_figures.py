"""
Create publication-quality figures and a markdown report for MedSAM CAMRI results.

Input:
    medsam_camri_rat_results.csv

Output directory:
    outputs/results_figures/

The script intentionally avoids pandas because the current medsam_env does not
include it. Keeping the dependency footprint small makes the plotting workflow
easy to rerun on the same environment used for MedSAM inference.

Run:
    MPLBACKEND=Agg medsam_env/bin/python create_medsam_results_figures.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = Path("medsam_camri_rat_results.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/results_figures")

METRICS = ["dice", "iou", "precision", "recall", "hausdorff_px"]
OVERLAP_METRICS = ["dice", "iou", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    # Keep the interface small: one input CSV and one output directory are enough
    # to regenerate all tables, figures, and the markdown report.
    parser = argparse.ArgumentParser(
        description="Create publication-style figures from MedSAM evaluation CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def set_publication_style() -> None:
    """Configure a restrained MICCAI/Medical Image Analysis-like style.

    The settings favor vector-friendly fonts, simple axes, and high-resolution
    exports so the same figures can be used in a draft manuscript or slide deck.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_results_csv(path: Path) -> list[dict[str, object]]:
    """Read the per-slice MedSAM metrics CSV into typed dictionaries.

    The evaluation CSV is slice-level, so every row remains independent here.
    Subject-level summaries are derived later rather than baked into the import.
    """

    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Store identifiers separately from metrics so numeric conversion is
            # explicit and downstream code does not rely on CSV strings.
            typed: dict[str, object] = {
                "subject_id": row["subject_id"],
                "slice_index": int(row["slice_index"]),
            }
            # Metrics may include NaN values for undefined Hausdorff distances,
            # so parse them through a helper instead of direct float conversion.
            for metric in METRICS:
                typed[metric] = parse_float(row.get(metric, "nan"))
            # Area and box columns are integer-valued even if a spreadsheet later
            # writes them with decimal formatting.
            for key in ["mask_area_gt", "mask_area_pred", "x_min", "y_min", "x_max", "y_max"]:
                typed[key] = int(float(row[key]))
            rows.append(typed)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def parse_float(value: str | None) -> float:
    """Parse floats while preserving blanks/NaN as np.nan."""

    # CSVs from different tools may represent missing values as blanks or text;
    # treating both as NaN keeps summary functions robust.
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def metric_array(rows: Iterable[dict[str, object]], metric: str) -> np.ndarray:
    """Return finite values for one metric as a NumPy array."""

    # Drop NaNs before plotting or summarizing so optional metrics do not poison
    # the rest of the report.
    values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save each figure as PNG for review and PDF for manuscript workflows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def compute_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Compute mean, std, median, min, and max for all slice-level metrics."""

    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        # All summaries are slice-level; subject-level aggregation is handled in
        # subject_summary() to avoid mixing units of analysis.
        values = metric_array(rows, metric)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(values.size),
        }
    return summary


def write_summary_csv(summary: dict[str, dict[str, float]], output_dir: Path) -> None:
    """Save summary statistics in machine-readable CSV form."""

    # Save the same numbers used in the figure table so manuscript values can be
    # audited without scraping text from an image.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary_statistics.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "median", "min", "max", "n"])
        for metric in METRICS:
            stats = summary[metric]
            writer.writerow(
                [
                    metric,
                    stats["mean"],
                    stats["std"],
                    stats["median"],
                    stats["min"],
                    stats["max"],
                    stats["n"],
                ]
            )


def make_summary_table(summary: dict[str, dict[str, float]], output_dir: Path) -> None:
    """Render mean +/- std summary statistics as a compact figure table."""

    table_rows = []
    for metric in METRICS:
        # Format to three decimals for figure readability; the companion CSV
        # retains full precision.
        stats = summary[metric]
        label = pretty_metric(metric)
        table_rows.append(
            [
                label,
                f"{stats['mean']:.3f} +/- {stats['std']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
            ]
        )

    # A table figure is convenient for slides and supplements where the markdown
    # report may not be visible.
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Metric", "Mean +/- SD", "Median", "Min", "Max"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        # Use a dark header and subtle banding so the table remains legible after
        # export to PDF or insertion into a manuscript.
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F6F8FA")
    ax.set_title("MedSAM Oracle-Box Segmentation Performance", pad=10, weight="bold")
    save_figure(fig, output_dir, "summary_statistics_table")


def make_bar_chart(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Plot mean overlap metrics with standard deviation error bars.

    Error bars show slice-level variability, not confidence intervals. This is
    useful here because failures tend to cluster in small edge slices.
    """

    # Restrict the bar chart to overlap-style metrics on the same [0, 1] scale.
    means = [np.mean(metric_array(rows, metric)) for metric in OVERLAP_METRICS]
    stds = [np.std(metric_array(rows, metric), ddof=1) for metric in OVERLAP_METRICS]
    labels = [pretty_metric(metric) for metric in OVERLAP_METRICS]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#72B7B2", "#59A14F", "#F28E2B"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#1F2937", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Mean Segmentation Metrics Across Slices")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    for i, mean in enumerate(means):
        # Place labels inside bars so long SD error bars do not collide with the
        # title area.
        ax.text(
            i,
            max(mean - 0.055, 0.03),
            f"{mean:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
    save_figure(fig, output_dir, "bar_mean_overlap_metrics")


def make_metric_boxplots(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Create boxplots for Dice, IoU, and Hausdorff distance."""

    # Dice/IoU share a bounded score axis; Hausdorff is in pixels and therefore
    # gets its own y-scale.
    specs = [
        ("dice", "Dice", "Score", (0, 1.05)),
        ("iou", "IoU", "Score", (0, 1.05)),
        ("hausdorff_px", "Hausdorff distance", "Pixels", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8))
    for ax, (metric, title, ylabel, ylim) in zip(axes, specs):
        # Show fliers because small edge-slice failures are scientifically useful
        # rather than plotting noise.
        values = metric_array(rows, metric)
        bp = ax.boxplot(
            values,
            widths=0.45,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#111827", "linewidth": 1.4},
            boxprops={"facecolor": "#A7C7E7", "edgecolor": "#1F2937", "linewidth": 0.8},
            whiskerprops={"color": "#1F2937", "linewidth": 0.8},
            capprops={"color": "#1F2937", "linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2.4, "markerfacecolor": "#9CA3AF", "markeredgewidth": 0},
        )
        _ = bp
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.suptitle("Distribution of Slice-Level Metrics", y=1.03, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "boxplots_dice_iou_hausdorff")


def make_dice_histogram(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Plot a histogram of slice-level Dice scores."""

    dice = metric_array(rows, "dice")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # Fixed bins over [0, 1] make histograms comparable across future benchmark
    # runs even if the score distribution changes.
    ax.hist(dice, bins=np.linspace(0, 1, 41), color="#4C78A8", edgecolor="white", linewidth=0.5)
    # Mark both mean and median because this distribution is skewed by difficult
    # edge slices.
    ax.axvline(np.mean(dice), color="#D62728", linewidth=1.4, label=f"Mean={np.mean(dice):.3f}")
    ax.axvline(np.median(dice), color="#111827", linewidth=1.4, linestyle="--", label=f"Median={np.median(dice):.3f}")
    ax.set_xlabel("Dice coefficient")
    ax.set_ylabel("Number of slices")
    ax.set_title("Distribution of Dice Scores")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    save_figure(fig, output_dir, "histogram_dice_scores")


def make_area_scatter(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Scatter plot of ground-truth mask area against Dice score.

    The trend line is clipped to [0, 1] because Dice is bounded; otherwise a
    least-squares fit can visually imply impossible scores above 1.
    """

    # Ground-truth area is a proxy for slice difficulty: very small masks often
    # correspond to anterior/posterior edge slices.
    area = np.array([int(row["mask_area_gt"]) for row in rows], dtype=np.float64)
    dice = metric_array(rows, "dice")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(area, dice, s=13, color="#4C78A8", alpha=0.45, edgecolors="none")

    if area.size > 1:
        # The trend is descriptive only; it helps reveal whether poor Dice values
        # are concentrated in small-mask slices.
        coeffs = np.polyfit(area, dice, deg=1)
        x_fit = np.linspace(area.min(), area.max(), 200)
        y_fit = np.clip(coeffs[0] * x_fit + coeffs[1], 0.0, 1.0)
        ax.plot(x_fit, y_fit, color="#D62728", linewidth=1.3, label="Linear trend")
        corr = np.corrcoef(area, dice)[0, 1]
        ax.text(
            0.03,
            0.08,
            f"Pearson r={corr:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D5DD"},
        )

    ax.set_xlabel("Ground-truth mask area (pixels)")
    ax.set_ylabel("Dice coefficient")
    ax.set_title("Mask Size vs Dice")
    ax.set_ylim(0, 1.05)
    ax.grid(color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "scatter_gt_area_vs_dice")


def group_by_subject(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Group per-slice rows by subject ID."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        # Keep the original row dictionaries intact so any later subject-level
        # analysis can still access box coordinates and areas.
        grouped[str(row["subject_id"])].append(row)
    return dict(grouped)


def subject_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute per-subject Dice summaries from the slice-level table."""

    summaries = []
    for subject_id, subject_rows in group_by_subject(rows).items():
        # Subject summaries collapse multiple slices to one row, which is useful
        # for identifying subjects with systematically worse performance.
        dice = metric_array(subject_rows, "dice")
        summaries.append(
            {
                "subject_id": subject_id,
                "n_slices": len(subject_rows),
                "mean_dice": float(np.mean(dice)),
                "median_dice": float(np.median(dice)),
                "std_dice": float(np.std(dice, ddof=1)) if dice.size > 1 else 0.0,
                "min_dice": float(np.min(dice)),
                "max_dice": float(np.max(dice)),
            }
        )
    # Sort from worst to best so plotting and report tables emphasize failure
    # modes first.
    return sorted(summaries, key=lambda item: float(item["mean_dice"]))


def write_subject_summary_csv(summaries: list[dict[str, object]], output_dir: Path) -> None:
    """Save per-subject Dice summary table."""

    # This table supports subject-level review without reopening the larger
    # per-slice CSV.
    out_path = output_dir / "per_subject_dice_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "n_slices",
                "mean_dice",
                "median_dice",
                "std_dice",
                "min_dice",
                "max_dice",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)


def make_per_subject_dice_distribution(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Plot subject-level Dice distributions sorted by mean Dice.

    Sorting by mean Dice makes outlier subjects visible without relying on a
    separate table lookup.
    """

    grouped = group_by_subject(rows)
    order = [
        item["subject_id"]
        for item in sorted(
            subject_summary(rows),
            key=lambda item: float(item["mean_dice"]),
        )
    ]
    data = [metric_array(grouped[subject_id], "dice") for subject_id in order]
    means = [float(np.mean(values)) for values in data]

    # Scale width with subject count, but cap it so the figure remains usable in
    # reports and does not become an ultra-wide artifact.
    fig_width = max(9.0, min(18.0, len(order) * 0.11))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.boxplot(
        data,
        positions=np.arange(len(order)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 0.8},
        boxprops={"facecolor": "#D7E8F7", "edgecolor": "#4C78A8", "linewidth": 0.45},
        whiskerprops={"color": "#4C78A8", "linewidth": 0.45},
        capprops={"color": "#4C78A8", "linewidth": 0.45},
    )
    ax.plot(np.arange(len(order)), means, color="#D62728", linewidth=1.1, label="Subject mean")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Dice coefficient")
    ax.set_xlabel("Subjects sorted by mean Dice")
    ax.set_title("Per-Subject Dice Distribution")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)

    # Showing every subject label would be unreadable for 131 subjects, so the
    # axis samples labels while preserving the full distribution.
    tick_step = max(1, math.ceil(len(order) / 18))
    ticks = np.arange(0, len(order), tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([order[i] for i in ticks], rotation=45, ha="right")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, output_dir, "per_subject_dice_distribution")


def worst_rows(rows: list[dict[str, object]], n: int = 20) -> list[dict[str, object]]:
    """Return the N worst slices sorted by Dice ascending."""

    # Ranking by Dice gives a direct failure list for visual QC and downstream
    # debugging of edge cases.
    return sorted(rows, key=lambda row: float(row["dice"]))[:n]


def write_worst_cases_csv(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Save the worst 20 slices as a CSV table for follow-up QC."""

    out_path = output_dir / "worst_20_slices_by_dice.csv"
    fieldnames = [
        "rank",
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
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            # Add an explicit rank column while preserving the original metric
            # fields from the evaluation CSV.
            out_row = {"rank": rank}
            out_row.update(row)
            writer.writerow(out_row)


def make_worst_cases_table(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Render the worst 20 slices as a figure table."""

    table_rows = []
    for rank, row in enumerate(rows, start=1):
        # The figure table uses compact rounding; the CSV retains full precision
        # for methods sections and supplementary materials.
        table_rows.append(
            [
                rank,
                row["subject_id"],
                row["slice_index"],
                f"{float(row['dice']):.3f}",
                f"{float(row['iou']):.3f}",
                f"{float(row['precision']):.3f}",
                f"{float(row['recall']):.3f}",
                f"{float(row['hausdorff_px']):.1f}" if np.isfinite(float(row["hausdorff_px"])) else "NA",
            ]
        )

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Rank", "Subject", "Slice", "Dice", "IoU", "Precision", "Recall", "HD px"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        # Red header styling signals that this table is a failure-mode summary,
        # not the main performance table.
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#7F1D1D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F9FAFB")
    ax.set_title("Worst 20 Slices Ranked by Dice", pad=12, weight="bold")
    save_figure(fig, output_dir, "worst_20_slices_by_dice")


def pretty_metric(metric: str) -> str:
    """Human-readable metric labels."""

    labels = {
        "dice": "Dice",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Recall",
        "hausdorff_px": "Hausdorff (px)",
    }
    return labels.get(metric, metric)


def write_markdown_report(
    rows: list[dict[str, object]],
    summary: dict[str, dict[str, float]],
    subject_summaries: list[dict[str, object]],
    worst: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Create a concise markdown report with key statistics and figure links."""

    # Derive headline statistics from the same row list used for figures so the
    # report cannot drift from the generated plots.
    subjects = sorted({str(row["subject_id"]) for row in rows})
    best_slice = max(rows, key=lambda row: float(row["dice"]))
    worst_slice = min(rows, key=lambda row: float(row["dice"]))
    best_subject = max(subject_summaries, key=lambda row: float(row["mean_dice"]))
    worst_subject = min(subject_summaries, key=lambda row: float(row["mean_dice"]))

    # Build the markdown report as a list of lines to keep formatting explicit
    # and easy to audit.
    lines = [
        "# MedSAM CAMRI Rat Oracle-Box Results",
        "",
        "## Dataset Summary",
        "",
        f"- Subjects evaluated: **{len(subjects)}**",
        f"- Slices evaluated: **{len(rows)}**",
        "",
        "## Metric Summary",
        "",
        "| Metric | Mean +/- SD | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        # Use four decimals in the report for manuscript-style numeric detail.
        stats = summary[metric]
        lines.append(
            f"| {pretty_metric(metric)} | "
            f"{stats['mean']:.4f} +/- {stats['std']:.4f} | "
            f"{stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Best and Worst Cases",
            "",
            (
                f"- Best slice: **{best_slice['subject_id']}**, slice "
                f"**{best_slice['slice_index']}**, Dice **{float(best_slice['dice']):.4f}**"
            ),
            (
                f"- Worst slice: **{worst_slice['subject_id']}**, slice "
                f"**{worst_slice['slice_index']}**, Dice **{float(worst_slice['dice']):.4f}**"
            ),
            (
                f"- Best subject by mean Dice: **{best_subject['subject_id']}**, "
                f"mean Dice **{float(best_subject['mean_dice']):.4f}**"
            ),
            (
                f"- Worst subject by mean Dice: **{worst_subject['subject_id']}**, "
                f"mean Dice **{float(worst_subject['mean_dice']):.4f}**"
            ),
            "",
            "## Worst 20 Slices Ranked by Dice",
            "",
            "| Rank | Subject | Slice | Dice | IoU | Precision | Recall | Hausdorff px |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(worst, start=1):
        # Include the worst slices directly in markdown so a reader can identify
        # failure cases without opening the CSV.
        lines.append(
            f"| {rank} | {row['subject_id']} | {row['slice_index']} | "
            f"{float(row['dice']):.4f} | {float(row['iou']):.4f} | "
            f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['hausdorff_px']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `summary_statistics_table.png`",
            "- `bar_mean_overlap_metrics.png`",
            "- `boxplots_dice_iou_hausdorff.png`",
            "- `histogram_dice_scores.png`",
            "- `scatter_gt_area_vs_dice.png`",
            "- `per_subject_dice_distribution.png`",
            "- `worst_20_slices_by_dice.png`",
            "",
            "All figures are also saved as PDF files for manuscript workflows.",
            "",
        ]
    )

    (output_dir / "medsam_camri_rat_report.md").write_text("\n".join(lines))


def main() -> None:
    """Load metrics, generate figures/tables, and write the markdown report."""

    args = parse_args()
    # Apply style before any figure is created; Matplotlib copies rcParams into
    # figures at creation time.
    set_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.csv}")
    rows = read_results_csv(args.csv)
    print(f"Loaded {len(rows)} slices from {len({row['subject_id'] for row in rows})} subjects")

    # Compute all derived tables first, then pass those immutable summaries into
    # plotting and reporting functions.
    summary = compute_summary(rows)
    subject_summaries = subject_summary(rows)
    worst = worst_rows(rows, n=20)

    print("Writing summary tables and figures...")
    write_summary_csv(summary, args.output_dir)
    write_subject_summary_csv(subject_summaries, args.output_dir)
    write_worst_cases_csv(worst, args.output_dir)

    # Generate publication figures after CSV summaries so a partial failure still
    # leaves machine-readable outputs for debugging.
    make_summary_table(summary, args.output_dir)
    make_bar_chart(rows, args.output_dir)
    make_metric_boxplots(rows, args.output_dir)
    make_dice_histogram(rows, args.output_dir)
    make_area_scatter(rows, args.output_dir)
    make_per_subject_dice_distribution(rows, args.output_dir)
    make_worst_cases_table(worst, args.output_dir)

    write_markdown_report(rows, summary, subject_summaries, worst, args.output_dir)
    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
