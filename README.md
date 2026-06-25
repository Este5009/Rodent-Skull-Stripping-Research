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
├── README.md
├── requirements.txt
├── AGENTS.md
├── MEMORY.md
├── MedSAM/                         # local external checkout, not tracked
├── medsam_env/                     # local virtual environment, not tracked
├── scripts/
│   ├── core/
│   │   ├── evaluate_medsam_camri_rat.py
│   │   ├── create_medsam_results_figures.py
│   │   ├── display_camri_rat_pair.py
│   │   └── whitestripe_mouse_diagnostic.py
│   └── experimental/
│       ├── evaluate_medsam_phase1_sensitivity.py
│       ├── evaluate_medsam_failure_analysis.py
│       └── create_medsam_skull_stripped_example.py
└── outputs/
    ├── benchmarks/
    ├── camri_rat_examples/
    ├── results_figures/
    ├── sensitivity/
    ├── failure_analysis/
    │   ├── runs/
    │   ├── latest/
    │   └── comparison/
    └── demo_skull_stripping/
```

`scripts/core/` contains stable, validated scripts. New experiments and
presentation/demo utilities belong in `scripts/experimental/`.

---

## Local Data And Model Paths

The expected local dataset layout is one level above this repository:

```text
../Datasets/
├── Image_Database/
│   └── CAMRI Rat Brain MRI Data/
└── Mask_Database/
    └── RodentBrainMask/
        └── CAMRI Rat/
```

The MedSAM checkpoint is expected at:

```text
MedSAM/work_dir/MedSAM/medsam_vit_b.pth
```

Raw imaging data, masks, checkpoints, external repositories, and virtual
environments are local-only artifacts and should not be committed.

---

## Environment

Create and activate the local environment:

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

Outputs are organized under:

```text
outputs/failure_analysis/
├── runs/<run-name>/
│   ├── tables/
│   ├── figures/
│   └── overlays/
├── latest/
└── comparison/
```

---

## Skull-Stripping Demo

Generate a complete MedSAM skull-stripped NIfTI demo volume for 3D Slicer:

```bash
./medsam_env/bin/python scripts/experimental/create_medsam_skull_stripped_example.py \
  --subject-id sub-001 \
  --device cpu
```

Outputs are saved to:

```text
outputs/demo_skull_stripping/sub-001/
├── volumes/
│   ├── medsam_pred_mask.nii.gz
│   └── medsam_skull_stripped.nii.gz
├── figures/
├── tables/
└── README.md
```

Both NIfTI volumes preserve the original MRI affine/header geometry and can be
opened directly in 3D Slicer.

---

## Notes

- This repository is a research prototype, not clinical software.
- Do not modify `scripts/core/` unless explicitly requested.
- Prefer incremental experiments in `scripts/experimental/`.
- Preserve raw data, masks, checkpoints, and generated local environments
  outside version control.
