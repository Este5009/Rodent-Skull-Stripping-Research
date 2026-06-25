# MedSAM Rodent MRI Segmentation Research

Research prototype for understanding how MedSAM behaves on biomedical MRI
segmentation tasks. The current benchmark uses CAMRI rat MRI and expert masks to
study why oracle-box MedSAM can underperform rodent-specific supervised models.

MedSAM is treated as one candidate backend inside a broader autonomous
segmentation pipeline. The present phase focuses on analysis, validation, and
failure modes before introducing new model architectures.

---

## Current Question

The current CAMRI rat benchmark asks:

```text
Why does MedSAM with expert-mask oracle boxes underperform stronger
rodent-specific supervised models?
```

Priority analyses:

- box-margin sensitivity
- preprocessing sensitivity
- implementation checks
- metric analysis
- visual QC and failure interpretation

---

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- docs/
|-- scripts/
|   |-- core/
|   `-- experimental/
`-- outputs/
    |-- benchmarks/
    |-- camri_rat_examples/
    |-- demo_skull_stripping/
    |-- diagnostics/
    |-- failure_analysis/
    |-- results_figures/
    |-- sensitivity/
    `-- smoke_tests/
```

`scripts/core/` contains stable, validated scripts. New experiments and
presentation/demo utilities belong in `scripts/experimental/`.

---

## Datasets

Datasets may be stored locally. The preferred layout is one level above this
repository:

```text
../Datasets/
|-- Image_Database/
|   `-- CAMRI Rat Brain MRI Data/
`-- Mask_Database/
    `-- RodentBrainMask/
        `-- CAMRI Rat/
```

Some older or local setups may instead use:

```text
Image Database/
Mask_Database/
```

Download sources:

- Expert ground-truth masks / RodentBrainMasks:
  https://drive.google.com/drive/folders/1cTlFFGL9iTUoZOT5Rgqi2ZAyqyPlXYd-
- CAMRI Rat MRI images:
  https://openneuro.org/datasets/ds002870/versions/1.0.0

---

## Local Files Not Tracked By Git

The following files and directories are local dependencies or large artifacts and
should remain outside version control:

- `MedSAM/` external checkout
- `medsam_env/` virtual environment, or an optional conda environment
- `MedSAM/work_dir/MedSAM/medsam_vit_b.pth` checkpoint
- raw MRI image datasets
- expert mask datasets
- large generated intermediates

---

## Environment

Create and activate a local environment:

```bash
python -m venv medsam_env
source medsam_env/bin/activate
pip install -r requirements.txt
```

Run project scripts with:

```bash
./medsam_env/bin/python <script>
```

---

## Script Overview

Stable scripts:

- `scripts/core/evaluate_medsam_camri_rat.py` - reference CAMRI rat benchmark
- `scripts/core/create_medsam_results_figures.py` - benchmark figures and summary
  tables
- `scripts/core/display_camri_rat_pair.py` - visual inspection of paired MRI/mask
  slices
- `scripts/core/whitestripe_mouse_diagnostic.py` - WhiteStripe preprocessing
  diagnostic

Experimental scripts:

- `scripts/experimental/evaluate_medsam_phase1_sensitivity.py` - Phase 1
  sensitivity experiments
- `scripts/experimental/evaluate_medsam_failure_analysis.py` - easy/hard subject
  failure analysis
- `scripts/experimental/create_medsam_skull_stripped_example.py` - NIfTI
  skull-stripping demo output

---

## Core Benchmark

Reference implementation:

```bash
./medsam_env/bin/python scripts/core/evaluate_medsam_camri_rat.py \
  --mri-root "../Datasets/Image_Database/CAMRI Rat Brain MRI Data" \
  --mask-root "../Datasets/Mask_Database/RodentBrainMask/CAMRI Rat"
```

The core benchmark matches CAMRI MRI volumes with expert masks, creates an
oracle box from each non-empty expert-mask slice, runs MedSAM, and reports Dice,
IoU, precision, recall, and Hausdorff.

---

## Failure Analysis

Use the experimental failure-analysis script to run targeted subject groups and
compare easy vs hard runs:

```bash
./medsam_env/bin/python scripts/experimental/evaluate_medsam_failure_analysis.py \
  --run-name easy_subjects \
  --subjects sub-001 sub-002 sub-003 sub-004 sub-005 \
  --device cpu

./medsam_env/bin/python scripts/experimental/evaluate_medsam_failure_analysis.py \
  --run-name hard_subjects \
  --subjects sub-050 sub-066 sub-086 sub-109 sub-112 \
  --device cpu

./medsam_env/bin/python scripts/experimental/evaluate_medsam_failure_analysis.py \
  --compare-runs easy_subjects hard_subjects
```

Outputs are organized under `outputs/failure_analysis/`, including run-specific
tables, figures, overlays, latest results, and comparison summaries.

---

## Skull-Stripping Demo

Generate a complete MedSAM skull-stripped NIfTI demo volume for 3D Slicer:

```bash
./medsam_env/bin/python scripts/experimental/create_medsam_skull_stripped_example.py \
  --subject-id sub-001 \
  --device cpu
```

Outputs are saved under `outputs/demo_skull_stripping/<subject-id>/`, including
NIfTI volumes, figures, tables, and a per-demo README. The generated NIfTI
volumes preserve the original MRI affine/header geometry and can be opened
directly in 3D Slicer.

---

## Notes

- This repository is a research prototype, not clinical software.
- Do not modify `scripts/core/` unless explicitly requested.
- Prefer incremental experiments in `scripts/experimental/`.
- Preserve raw data, masks, checkpoints, local environments, and large generated
  intermediates outside version control.
