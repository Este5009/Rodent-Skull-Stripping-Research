r"""Oracle feasibility test for coarse shape-aware MedSAM mask prompts.

The expert mask is deliberately degraded through a configurable low-resolution
bottleneck before it reaches MedSAM's mask prompt encoder. Four methods share
the same MRI preprocessing, image embedding, decoder, threshold, and metrics:

1. Single oracle box.
2. Raw oracle component boxes, independently decoded and unioned.
3. Coarse mask prompt only.
4. Raw component boxes plus the same coarse mask prompt.

Example verification:
    medsam_env\Scripts\python.exe scripts\experimental\evaluate_medsam_mask_prompt_oracle.py \
        --subjects sub-086 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TEMP", ".")) / "medsam_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.getenv("TEMP", ".")) / "medsam_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure, morphology, transform


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
DEFAULT_OUTPUT_DIR = Path("outputs/mask_prompt_oracle")
DEFAULT_COARSE_SIZE = 32
DEFAULT_PERTURBATION_RADIUS = 1
DEFAULT_MIN_COMPONENT_AREA = 10
DEFAULT_MAX_EXAMPLES = 10
MASK_INPUT_SIZE = 256
METHODS = (
    ("single", "Single oracle box"),
    ("multi", "Raw multi-box"),
    ("mask_only", "Coarse mask only"),
    ("box_mask", "Multi-box + coarse mask"),
)
METRICS = ("dice", "iou", "precision", "recall", "hausdorff_px", "mask_area_pred")


@dataclass
class CoarsePrompt:
    """Soft coarse prompt probabilities and safe mask-input logits."""

    probabilities: np.ndarray
    logits: np.ndarray


@dataclass
class EvaluatedSlice:
    """Arrays and metrics retained for qualitative example selection."""

    subject_id: str
    slice_index: int
    image_uint8: np.ndarray
    gt_mask: np.ndarray
    coarse_probabilities: np.ndarray
    predictions: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float]]

    @property
    def mask_prompt_improvement(self) -> float:
        """Best mask-prompt Dice minus the best rectangle-only Dice."""

        best_box = max(self.metrics["single"]["dice"], self.metrics["multi"]["dice"])
        best_mask = max(
            self.metrics["mask_only"]["dice"], self.metrics["box_mask"]["dice"]
        )
        return float(best_mask - best_box)


def parse_args() -> argparse.Namespace:
    """Read experiment and output options."""

    parser = argparse.ArgumentParser(
        description="Compare oracle boxes with deliberately coarse MedSAM mask prompts."
    )
    parser.add_argument("--mri-root", type=Path, default=MRI_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--coarse-size", type=int, default=DEFAULT_COARSE_SIZE)
    parser.add_argument(
        "--perturbation",
        choices=["none", "erode", "dilate"],
        default="none",
    )
    parser.add_argument(
        "--perturbation-radius", type=int, default=DEFAULT_PERTURBATION_RADIUS
    )
    parser.add_argument("--box-margin", type=int, default=BOX_MARGIN)
    parser.add_argument(
        "--min-component-area", type=int, default=DEFAULT_MIN_COMPONENT_AREA
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    args = parser.parse_args()

    if not 2 <= args.coarse_size < MASK_INPUT_SIZE:
        parser.error(f"--coarse-size must be between 2 and {MASK_INPUT_SIZE - 1}")
    if args.perturbation_radius < 0:
        parser.error("--perturbation-radius must be non-negative")
    if args.box_margin < 0:
        parser.error("--box-margin must be non-negative")
    if args.min_component_area < 1:
        parser.error("--min-component-area must be at least 1")
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    return args


def select_subjects(
    mri_root: Path, mask_root: Path, requested: list[str]
) -> list[SubjectPair]:
    """Select matched MRI/mask pairs in the requested order."""

    pairs, warnings = match_subjects(mri_root, mask_root)
    for warning in warnings:
        print(f"Data warning: {warning.subject_id}: {warning.reason}")
    pair_by_id = {pair.subject_id: pair for pair in pairs}
    selected = [pair_by_id[subject] for subject in dict.fromkeys(requested) if subject in pair_by_id]
    missing = [subject for subject in dict.fromkeys(requested) if subject not in pair_by_id]
    if missing:
        print("Requested subjects not found: " + ", ".join(missing))
    if not selected:
        raise RuntimeError("None of the requested subjects have matched MRI and mask files.")
    return selected


def non_empty_slice_indices(mask_volume: np.ndarray) -> list[int]:
    """Return every slice containing expert-mask foreground."""

    return [
        index
        for index in range(mask_volume.shape[2])
        if np.any(mask_volume[:, :, index] > 0)
    ]


def component_boxes(
    gt_mask: np.ndarray, min_component_area: int, margin: int
) -> list[tuple[int, int, int, int]]:
    """Create fixed-margin boxes for retained 8-connected components."""

    labels = measure.label(gt_mask.astype(bool), connectivity=2)
    boxes: list[tuple[int, int, int, int]] = []
    for region in measure.regionprops(labels):
        if region.area < min_component_area:
            continue
        component = (labels == region.label).astype(np.uint8)
        box = bbox_from_mask(component, margin=margin)
        if box is not None:
            boxes.append(box)
    return boxes


def generate_coarse_prompt(
    gt_mask: np.ndarray,
    coarse_size: int,
    perturbation: str,
    perturbation_radius: int,
) -> CoarsePrompt:
    """Degrade an expert mask through a low-resolution bottleneck and return logits."""

    coarse = transform.resize(
        gt_mask.astype(np.float32),
        (coarse_size, coarse_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    if perturbation != "none" and perturbation_radius > 0:
        footprint = morphology.disk(perturbation_radius)
        coarse_binary = coarse >= 0.5
        if perturbation == "erode":
            coarse = morphology.binary_erosion(
                coarse_binary, footprint=footprint
            ).astype(np.float32)
        else:
            coarse = morphology.binary_dilation(
                coarse_binary, footprint=footprint
            ).astype(np.float32)

    probabilities = transform.resize(
        coarse,
        (MASK_INPUT_SIZE, MASK_INPUT_SIZE),
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    logits = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
    if not np.isfinite(logits).all():
        raise RuntimeError("Coarse mask prompt contains non-finite logits.")
    return CoarsePrompt(probabilities=probabilities, logits=logits)


@torch.no_grad()
def encode_image(
    model: torch.nn.Module, image_uint8: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Create the validated MedSAM 1024×1024 image embedding once per slice."""

    image_rgb = np.repeat(image_uint8[:, :, None], 3, axis=-1)
    image_1024 = transform.resize(
        image_rgb,
        (1024, 1024),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    image_1024 = (image_1024 - image_1024.min()) / np.clip(
        image_1024.max() - image_1024.min(), a_min=1e-8, a_max=None
    )
    image_tensor = (
        torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    return model.image_encoder(image_tensor)


def box_tensor(
    box: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    """Scale one native-image box to MedSAM's 1024 coordinate system."""

    if box is None:
        return None
    height, width = image_shape
    scaled = np.asarray([box], dtype=np.float32)
    scaled = scaled / np.asarray([width, height, width, height]) * 1024
    return torch.as_tensor(scaled, dtype=torch.float32, device=device)[:, None, :]


@torch.no_grad()
def decode_prompt(
    model: torch.nn.Module,
    image_embedding: torch.Tensor,
    output_shape: tuple[int, int],
    device: torch.device,
    box: tuple[int, int, int, int] | None = None,
    mask_logits: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one box/mask prompt using the validated MedSAM thresholding path."""

    mask_tensor = None
    if mask_logits is not None:
        mask_tensor = torch.as_tensor(
            mask_logits[None, None], dtype=torch.float32, device=device
        )
        if tuple(mask_tensor.shape) != (1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE):
            raise ValueError(f"Unexpected mask prompt shape: {tuple(mask_tensor.shape)}")
        if not torch.isfinite(mask_tensor).all():
            raise ValueError("Mask prompt tensor contains non-finite logits.")

    sparse, dense = model.prompt_encoder(
        points=None,
        boxes=box_tensor(box, output_shape, device),
        masks=mask_tensor,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    probabilities = F.interpolate(
        torch.sigmoid(low_res_logits),
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )
    return (probabilities.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def predict_union(
    model: torch.nn.Module,
    image_embedding: torch.Tensor,
    image_shape: tuple[int, int],
    device: torch.device,
    boxes: list[tuple[int, int, int, int] | None],
    mask_logits: np.ndarray | None = None,
) -> np.ndarray:
    """Decode prompts independently and union their binary predictions."""

    union = np.zeros(image_shape, dtype=np.uint8)
    for box in boxes:
        prediction = decode_prompt(
            model,
            image_embedding,
            image_shape,
            device,
            box=box,
            mask_logits=mask_logits,
        )
        union = np.logical_or(union, prediction).astype(np.uint8)
    return union


def metric_columns(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Flatten one method's core metrics into a comparison row."""

    return {f"{prefix}_{metric}": metrics[metric] for metric in METRICS}


def validate_preflight(
    pair: SubjectPair,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Validate prompt safety and exact box-only equivalence before full evaluation."""

    mri = load_nifti_data(pair.mri_path)
    gt_volume = load_nifti_data(pair.mask_path) > 0
    slice_index = non_empty_slice_indices(gt_volume)[0]
    gt_mask = gt_volume[:, :, slice_index].astype(np.uint8)
    image_uint8 = percentile_to_uint8(mri[:, :, slice_index])
    prompt = generate_coarse_prompt(
        gt_mask, args.coarse_size, args.perturbation, args.perturbation_radius
    )
    prompt_tensor = torch.as_tensor(prompt.logits[None, None])
    if tuple(prompt_tensor.shape) != (1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE):
        raise RuntimeError(f"Mask prompt preflight shape failed: {prompt_tensor.shape}")
    if not torch.isfinite(prompt_tensor).all():
        raise RuntimeError("Mask prompt preflight found non-finite logits.")

    exact_256 = transform.resize(
        gt_mask.astype(np.float32),
        (MASK_INPUT_SIZE, MASK_INPUT_SIZE),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    if np.array_equal(prompt.probabilities, exact_256):
        raise RuntimeError("Coarse prompt unexpectedly equals the exact expert mask.")

    box = bbox_from_mask(gt_mask, margin=args.box_margin)
    if box is None:
        raise RuntimeError("Preflight slice did not produce an oracle box.")
    image_embedding = encode_image(model, image_uint8, device)
    shared_prediction = decode_prompt(
        model, image_embedding, image_uint8.shape, device, box=box
    )
    core_prediction = medsam_official_inference(model, image_uint8, box, device)
    if not np.array_equal(shared_prediction, core_prediction):
        disagreement = int(np.count_nonzero(shared_prediction != core_prediction))
        raise RuntimeError(
            f"Single-box path disagrees with core inference at {disagreement} pixels."
        )

    dice = segmentation_metrics(shared_prediction, gt_mask)["dice"]
    print("Preflight checks:")
    print(f"  mask tensor shape: {tuple(prompt_tensor.shape)} [PASS]")
    print("  finite mask logits: PASS")
    print("  coarse prompt differs from exact expert mask: PASS")
    print(f"  single-box equals core baseline: PASS (Dice={dice:.6f})")


def evaluate_subject(
    pair: SubjectPair,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[EvaluatedSlice]]:
    """Evaluate all non-empty slices with identical image/decoder conventions."""

    mri = load_nifti_data(pair.mri_path)
    gt_volume = load_nifti_data(pair.mask_path) > 0
    if mri.shape != gt_volume.shape or mri.ndim != 3:
        raise ValueError(
            f"{pair.subject_id}: incompatible MRI {mri.shape} and mask {gt_volume.shape}"
        )

    indices = non_empty_slice_indices(gt_volume)
    rows: list[dict[str, object]] = []
    records: list[EvaluatedSlice] = []
    for position, slice_index in enumerate(indices, start=1):
        print(f"  [{position:03d}/{len(indices):03d}] slice {slice_index:03d}", flush=True)
        gt_mask = gt_volume[:, :, slice_index].astype(np.uint8)
        image_uint8 = percentile_to_uint8(mri[:, :, slice_index])
        single_box = bbox_from_mask(gt_mask, margin=args.box_margin)
        if single_box is None:
            continue
        multi_boxes = component_boxes(
            gt_mask, args.min_component_area, args.box_margin
        )
        coarse = generate_coarse_prompt(
            gt_mask, args.coarse_size, args.perturbation, args.perturbation_radius
        )
        embedding = encode_image(model, image_uint8, device)

        predictions = {
            "single": predict_union(
                model, embedding, image_uint8.shape, device, [single_box]
            ),
            "multi": predict_union(
                model, embedding, image_uint8.shape, device, multi_boxes
            ),
            "mask_only": predict_union(
                model,
                embedding,
                image_uint8.shape,
                device,
                [None],
                mask_logits=coarse.logits,
            ),
            "box_mask": predict_union(
                model,
                embedding,
                image_uint8.shape,
                device,
                multi_boxes,
                mask_logits=coarse.logits,
            ),
        }
        metrics = {
            method: segmentation_metrics(prediction, gt_mask)
            for method, prediction in predictions.items()
        }
        component_count = len(measure.regionprops(measure.label(gt_mask, connectivity=2)))
        row: dict[str, object] = {
            "subject_id": pair.subject_id,
            "slice_index": slice_index,
            "component_count": component_count,
            "retained_component_boxes": len(multi_boxes),
            "gt_area": int(gt_mask.sum()),
            "coarse_size": args.coarse_size,
            "perturbation": args.perturbation,
            "perturbation_radius": args.perturbation_radius,
        }
        for method, _ in METHODS:
            row.update(metric_columns(method, metrics[method]))
        best_box = max(metrics["single"]["dice"], metrics["multi"]["dice"])
        best_mask = max(metrics["mask_only"]["dice"], metrics["box_mask"]["dice"])
        row.update(
            {
                "multi_over_single_dice": metrics["multi"]["dice"]
                - metrics["single"]["dice"],
                "mask_only_over_single_dice": metrics["mask_only"]["dice"]
                - metrics["single"]["dice"],
                "box_mask_over_multi_dice": metrics["box_mask"]["dice"]
                - metrics["multi"]["dice"],
                "best_mask_over_best_box_dice": best_mask - best_box,
            }
        )
        rows.append(row)
        records.append(
            EvaluatedSlice(
                subject_id=pair.subject_id,
                slice_index=slice_index,
                image_uint8=image_uint8,
                gt_mask=gt_mask,
                coarse_probabilities=coarse.probabilities,
                predictions=predictions,
                metrics=metrics,
            )
        )
    return rows, records


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write a non-empty list of flat dictionaries."""

    rows = list(rows)
    if not rows:
        raise RuntimeError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_subjects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create one mean-metric comparison row per subject."""

    summaries: list[dict[str, object]] = []
    for subject_id in dict.fromkeys(str(row["subject_id"]) for row in rows):
        selected = [row for row in rows if row["subject_id"] == subject_id]
        summary: dict[str, object] = {
            "subject_id": subject_id,
            "slices": len(selected),
        }
        for method, _ in METHODS:
            for metric in METRICS:
                values = np.asarray(
                    [float(row[f"{method}_{metric}"]) for row in selected]
                )
                finite_values = values[np.isfinite(values)]
                summary[f"{method}_mean_{metric}"] = (
                    float(finite_values.mean())
                    if finite_values.size > 0
                    else float("nan")
                )
        improvements = np.asarray(
            [float(row["best_mask_over_best_box_dice"]) for row in selected]
        )
        summary["mean_best_mask_improvement"] = float(improvements.mean())
        summary["mask_improved_slices"] = int(np.count_nonzero(improvements > 0))
        summary["mask_worsened_slices"] = int(np.count_nonzero(improvements < 0))
        summaries.append(summary)
    return summaries


def summarize_methods(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate each prompt method across all evaluated slices."""

    single_mean = float(np.mean([float(row["single_dice"]) for row in rows]))
    summaries: list[dict[str, object]] = []
    for method, label in METHODS:
        dice = np.asarray([float(row[f"{method}_dice"]) for row in rows])
        summaries.append(
            {
                "method": label,
                "slices": len(rows),
                "mean_dice": float(dice.mean()),
                "median_dice": float(np.median(dice)),
                "std_dice": float(dice.std()),
                "mean_iou": float(
                    np.mean([float(row[f"{method}_iou"]) for row in rows])
                ),
                "mean_precision": float(
                    np.mean([float(row[f"{method}_precision"]) for row in rows])
                ),
                "mean_recall": float(
                    np.mean([float(row[f"{method}_recall"]) for row in rows])
                ),
                "mean_dice_change_vs_single": float(dice.mean() - single_mean),
            }
        )
    return summaries


def write_presentation_summary(
    path: Path,
    rows: list[dict[str, object]],
    method_summary: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    """Write a concise presentation-ready result summary."""

    improvements = np.asarray(
        [float(row["best_mask_over_best_box_dice"]) for row in rows]
    )
    method_lines = "\n".join(
        f"- {row['method']}: mean Dice {float(row['mean_dice']):.4f}"
        for row in method_summary
    )
    best = max(method_summary, key=lambda row: float(row["mean_dice"]))
    markdown = f"""# Coarse Mask-Prompt Oracle Summary

## Question

Can a deliberately coarse, shape-aware mask prompt solve difficult MedSAM
failures better than oracle rectangular prompts?

## Run

- Subjects: {', '.join(dict.fromkeys(str(row['subject_id']) for row in rows))}
- Non-empty slices: {len(rows)}
- Coarse bottleneck: {args.coarse_size} x {args.coarse_size}
- Perturbation: {args.perturbation}, radius {args.perturbation_radius}

## Results

{method_lines}

- Best mean method: {best['method']} ({float(best['mean_dice']):.4f})
- Mean best-mask improvement over best-box: {float(improvements.mean()):+.4f}
- Improved / worsened / unchanged slices: {int((improvements > 0).sum())} / {int((improvements < 0).sum())} / {int((improvements == 0).sum())}

## Interpretation

This is an oracle feasibility result. The prompt has passed through a
low-resolution bottleneck and is not an exact expert boundary, but an automatic
proposal network must still learn to generate comparable coarse shapes.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def rows_by_subject(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group rows by subject and sort by slice index."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)
    for selected in grouped.values():
        selected.sort(key=lambda row: int(row["slice_index"]))
    return grouped


def save_subject_figures(
    rows: list[dict[str, object]], output_dir: Path
) -> None:
    """Save four-method Dice and improvement lines for every subject."""

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for subject_id, selected in rows_by_subject(rows).items():
        slices = np.asarray([int(row["slice_index"]) for row in selected])
        figure, ax = plt.subplots(figsize=(11, 5))
        for (method, label), color in zip(METHODS, colors):
            dice = [float(row[f"{method}_dice"]) for row in selected]
            ax.plot(slices, dice, "o-", linewidth=1.3, markersize=4, label=label, color=color)
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
            output_dir / f"dice_by_slice_{subject_id}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        figure, ax = plt.subplots(figsize=(11, 4.5))
        ax.axhline(0, color="black", linewidth=1)
        comparisons = (
            ("multi_over_single_dice", "Raw multi-box - single", "tab:orange"),
            ("mask_only_over_single_dice", "Mask only - single", "tab:green"),
            ("box_mask_over_multi_dice", "Box + mask - raw multi-box", "tab:purple"),
        )
        for column, label, color in comparisons:
            values = [float(row[column]) for row in selected]
            ax.plot(slices, values, "o-", linewidth=1.3, markersize=4, label=label, color=color)
        ax.set(
            xlabel="Slice index",
            ylabel="Dice improvement",
            title=f"{subject_id}: prompt-method improvement by slice",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"improvement_by_slice_{subject_id}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_method_comparison(
    rows: list[dict[str, object]], output_path: Path
) -> None:
    """Save an aggregate four-method Dice comparison."""

    labels = [label for _, label in METHODS]
    values = [
        np.asarray([float(row[f"{method}_dice"]) for row in rows])
        for method, _ in METHODS
    ]
    means = [float(value.mean()) for value in values]
    standard_deviations = [float(value.std()) for value in values]
    figure, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=standard_deviations, capsize=5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.3f}", ha="center", va="bottom")
    ax.set(ylabel="Mean Dice", ylim=(0, 1), title="MedSAM oracle prompt-method comparison")
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def show_mask(ax: plt.Axes, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Show an MRI with a translucent binary mask."""

    ax.imshow(image, cmap="gray")
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.45)
    if np.any(mask):
        ax.contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
    ax.set_title(title)
    ax.axis("off")


def show_error_map(
    ax: plt.Axes, gt_mask: np.ndarray, prediction: np.ndarray, title: str
) -> None:
    """Show false positives in red and false negatives in blue."""

    gt = gt_mask.astype(bool)
    pred = prediction.astype(bool)
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[np.logical_and(gt, pred)] = (0.72, 0.72, 0.72)
    rgb[np.logical_and(~gt, pred)] = (1.0, 0.12, 0.12)
    rgb[np.logical_and(gt, ~pred)] = (0.12, 0.42, 1.0)
    ax.imshow(rgb)
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(color=(1.0, 0.12, 0.12), label="False positive"),
            Patch(color=(0.12, 0.42, 1.0), label="False negative"),
            Patch(color=(0.72, 0.72, 0.72), label="True positive"),
        ],
        loc="lower left",
        fontsize=7,
    )
    ax.axis("off")


def save_example(record: EvaluatedSlice, path: Path) -> None:
    """Save the required eight-panel qualitative comparison."""

    figure, axes_grid = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes_grid.ravel()
    axes[0].imshow(record.image_uint8, cmap="gray")
    axes[0].set_title("MRI")
    axes[0].axis("off")
    show_mask(axes[1], record.image_uint8, record.gt_mask, "Expert mask")
    axes[2].imshow(record.coarse_probabilities, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Generated coarse mask prompt")
    axes[2].axis("off")
    panel_methods = ("single", "multi", "mask_only", "box_mask")
    for axis, method in zip(axes[3:7], panel_methods):
        label = dict(METHODS)[method]
        dice = record.metrics[method]["dice"]
        show_mask(
            axis,
            record.image_uint8,
            record.predictions[method],
            f"{label}\nDice={dice:.3f}",
        )
    best_method = max(panel_methods, key=lambda method: record.metrics[method]["dice"])
    show_error_map(
        axes[7],
        record.gt_mask,
        record.predictions[best_method],
        f"Best FP / FN\n{dict(METHODS)[best_method]}",
    )
    figure.suptitle(
        f"{record.subject_id} | slice {record.slice_index} | "
        f"best mask improvement={record.mask_prompt_improvement:+.3f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_examples(
    records: list[EvaluatedSlice], output_dir: Path, maximum: int
) -> None:
    """Save largest improvements and truthful mask-prompt regressions."""

    ranked = sorted(records, key=lambda record: record.mask_prompt_improvement, reverse=True)
    for rank, record in enumerate(ranked[:maximum], start=1):
        filename = (
            f"{rank:02d}_{record.subject_id}_slice-{record.slice_index:03d}_"
            f"delta-{record.mask_prompt_improvement:+.3f}.png"
        )
        save_example(record, output_dir / "largest_improvements" / filename)

    worsened = sorted(
        (record for record in records if record.mask_prompt_improvement < 0),
        key=lambda record: record.mask_prompt_improvement,
    )
    for rank, record in enumerate(worsened[:maximum], start=1):
        filename = (
            f"{rank:02d}_{record.subject_id}_slice-{record.slice_index:03d}_"
            f"delta-{record.mask_prompt_improvement:+.3f}.png"
        )
        save_example(record, output_dir / "worsened" / filename)


def main() -> None:
    """Run preflight validation, inference, and organized reporting."""

    args = parse_args()
    device = choose_device(args.device)
    pairs = select_subjects(args.mri_root, args.mask_root, args.subjects)
    print("========== MedSAM Coarse Mask-Prompt Oracle ==========")
    print("Subjects:       " + ", ".join(pair.subject_id for pair in pairs))
    print(f"Device:         {device}")
    print(f"Coarse size:    {args.coarse_size} x {args.coarse_size}")
    print(f"Perturbation:   {args.perturbation} (radius={args.perturbation_radius})")
    print(f"Output:         {args.output_dir}")
    print("Loading MedSAM checkpoint...", flush=True)
    model = load_medsam_model(args.checkpoint, device)
    validate_preflight(pairs[0], model, device, args)

    all_rows: list[dict[str, object]] = []
    all_records: list[EvaluatedSlice] = []
    for pair in pairs:
        print(f"Evaluating {pair.subject_id}...", flush=True)
        rows, records = evaluate_subject(pair, model, device, args)
        all_rows.extend(rows)
        all_records.extend(records)
        print(f"  Completed {len(rows)} slices")
    if not all_rows:
        raise RuntimeError("No non-empty slices were evaluated.")

    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    method_summary = summarize_methods(all_rows)
    write_csv(tables / "per_slice_comparison.csv", all_rows)
    write_csv(tables / "per_subject_summary.csv", summarize_subjects(all_rows))
    write_csv(tables / "prompt_method_summary.csv", method_summary)
    write_presentation_summary(
        tables / "presentation_summary.md", all_rows, method_summary, args
    )
    save_subject_figures(all_rows, figures)
    save_method_comparison(all_rows, figures / "method_comparison.png")
    save_examples(all_records, args.output_dir / "examples", args.max_examples)

    print("========== Complete ==========")
    print(f"Subjects evaluated: {len(pairs)}")
    print(f"Slices evaluated:   {len(all_rows)}")
    for row in method_summary:
        print(f"{row['method']:<25} Dice={float(row['mean_dice']):.4f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
