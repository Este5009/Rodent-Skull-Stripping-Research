# Rodent Skull Stripping and Brain Segmentation Research

Research code for evaluating MedSAM-based skull stripping and brain segmentation on rodent MRI. The current main experiment benchmarks MedSAM on CAMRI rat brain MRI using oracle bounding boxes generated from expert manual brain masks.

This repository contains code, result summaries, and selected visual outputs. Raw MRI data, masks, model checkpoints, virtual environments, and external repositories are intentionally excluded.

---

## Project Goal

The main question is:

> If MedSAM is given an ideal 2D bounding box from an expert brain mask, how accurately can it segment rodent brain MRI slices?

This is an **oracle-box benchmark**, not a fully autonomous segmentation pipeline. The expert mask is used only to generate the bounding box prompt and to evaluate the predicted mask.

Workflow:

```text
MRI volume
    ↓
Expert brain mask
    ↓
Oracle box per non-empty slice
    ↓
MedSAM inference
    ↓
Predicted mask in memory
    ↓
Dice / IoU / precision / recall / Hausdorff
    ↓
CSV results and visual summaries
```

Predicted masks are used to compute metrics during runtime, but they are not currently saved as standalone mask files.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── evaluate_medsam_camri_rat.py
├── create_medsam_results_figures.py
├── display_camri_rat_pair.py
├── whitestripe_mouse_diagnostic.py
├── medsam_camri_rat_results.csv
└── outputs/
    ├── camri_rat_examples/
    ├── diagnostics/
    └── results_figures/
```

Ignored local folders/files include:

```text
medsam_env/
MedSAM/
Image Database/
Mask_Database/
Test_1_imgs/
Research Papers/
*.dcm
*.nii
*.nii.gz
*.pth
*.pt
*.ckpt
```

---

## Script Overview

### `evaluate_medsam_camri_rat.py`

Main MedSAM benchmark script.

It matches CAMRI MRI volumes with expert masks, generates oracle bounding boxes from non-empty mask slices, runs MedSAM inference, computes segmentation metrics, writes `medsam_camri_rat_results.csv`, and saves selected qualitative examples in `outputs/camri_rat_examples/`.

### `create_medsam_results_figures.py`

Result reporting script.

It reads `medsam_camri_rat_results.csv` and generates summary tables, plots, worst-case slice summaries, per-subject Dice summaries, and a markdown report in `outputs/results_figures/`.

This script does not run MedSAM.

### `display_camri_rat_pair.py`

Dataset inspection utility.

It displays a CAMRI MRI volume and its corresponding expert mask to verify that the files are matched, readable, and visually aligned before running the full benchmark.

### `whitestripe_mouse_diagnostic.py`

Preprocessing diagnostic utility.

It explores WhiteStripe-style intensity normalization for mouse MRI by plotting ROI histograms, candidate intensity peaks, and normalized previews. It is exploratory and not required for the main CAMRI benchmark.

---

## Key Outputs

### `medsam_camri_rat_results.csv`

Main quantitative results file. Each row corresponds to one evaluated 2D slice and includes:

* Subject ID
* Slice index
* Dice
* IoU
* Precision
* Recall
* Hausdorff distance
* Ground-truth mask area
* Predicted mask area
* Oracle bounding-box coordinates

### `outputs/camri_rat_examples/`

Selected visual examples from the benchmark. These figures show the MRI slice, oracle box, expert mask, and MedSAM prediction overlay.

### `outputs/results_figures/`

Summary figures and tables generated from the benchmark CSV.

### `outputs/diagnostics/`

Diagnostic figures from preprocessing experiments, such as WhiteStripe histogram checks.

---

## Local Setup

### 1. Create environment

macOS/Linux:

```bash
python -m venv medsam_env
source medsam_env/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv medsam_env
.\medsam_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 2. Clone MedSAM locally

The official MedSAM source code is not included in this repository.

Clone it inside the project root:

```bash
git clone https://github.com/bowang-lab/MedSAM.git
```

Expected local structure:

```text
.
├── MedSAM/
├── evaluate_medsam_camri_rat.py
└── ...
```

---

### 3. Add MedSAM checkpoint

Place the MedSAM ViT-B checkpoint at:

```text
MedSAM/work_dir/MedSAM/medsam_vit_b.pth
```

The checkpoint is not tracked by Git.

---

### 4. Add local datasets

The main benchmark expects:

```text
Image Database/
└── CAMRI Rat Brain MRI Data/

Mask_Database/
└── RodentBrainMask/
    └── CAMRI Rat/
```

These folders are ignored because they contain raw imaging data and masks.

---

## Running the Benchmark

Smoke test:

macOS/Linux:

```bash
MPLBACKEND=Agg medsam_env/bin/python evaluate_medsam_camri_rat.py --max-subjects 1
```

Windows PowerShell:

```powershell
$env:MPLBACKEND="Agg"
python evaluate_medsam_camri_rat.py --max-subjects 1
```

Full run:

macOS/Linux:

```bash
MPLBACKEND=Agg medsam_env/bin/python evaluate_medsam_camri_rat.py
```

Windows PowerShell:

```powershell
$env:MPLBACKEND="Agg"
python evaluate_medsam_camri_rat.py
```

---

## Generating Figures

After running the benchmark:

macOS/Linux:

```bash
MPLBACKEND=Agg medsam_env/bin/python create_medsam_results_figures.py
```

Windows PowerShell:

```powershell
$env:MPLBACKEND="Agg"
python create_medsam_results_figures.py
```

Outputs are saved to:

```text
outputs/results_figures/
```

---

## Collaboration Notes

Before working on another computer:

```bash
git pull
```

After making changes:

```bash
git add .
git commit -m "Describe the change"
git push
```

Do not commit raw data, masks, checkpoints, virtual environments, or external repositories.

---

## Author

Esteban Felix
Computer Engineering
Purdue University Indianapolis
