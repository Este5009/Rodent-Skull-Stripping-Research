# MedSAM Rodent Brain MRI Segmentation Evaluation

This repository contains an evaluation pipeline for MedSAM-based rodent brain MRI segmentation. The main experiment benchmarks MedSAM on CAMRI rat brain MRI using oracle bounding-box prompts generated from expert manual brain masks.

The project focuses on segmentation evaluation, visualization, and result reporting. Raw imaging datasets and model checkpoints are excluded from version control.

---

## Project Overview

The primary research question is:

> If MedSAM is given an ideal 2D bounding box derived from an expert brain mask, how accurately can it segment rodent brain MRI slices?

The benchmark workflow is:

```text
MRI volume
    ↓
Expert manual mask
    ↓
Oracle bounding box generation
    ↓
MedSAM inference
    ↓
Predicted brain mask
    ↓
Segmentation metrics
    ↓
Summary figures and reports
```

The evaluation computes:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- Hausdorff distance

---

## Repository Structure

```text
.
├── MedSAM/                              # Official MedSAM source code
├── scripts/                             # Project scripts
│   ├── evaluate_medsam_camri_rat.py
│   ├── create_medsam_results_figures.py
│   ├── display_camri_rat_pair.py
│   ├── medsam_display_test.py
│   └── whitestripe_mouse_diagnostic.py
│
├── results/                             # Selected result files
│   ├── medsam_camri_rat_results.csv
│   └── example result images
│
├── outputs/                             # Generated visualizations
│   ├── camri_rat_examples/
│   └── results_figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The following local folders are intentionally excluded:

```text
medsam_env/
Image Database/
Mask_Database/
Test_1_imgs/
Research Papers/
MedSAM/work_dir/
```

---

## External Files Not Included

This repository does not include:

- Raw DICOM files
- NIfTI MRI volumes
- Expert mask volumes
- MedSAM checkpoint files
- Local Python virtual environments
- Research papers or PDFs

The MedSAM checkpoint should be placed locally at:

```text
MedSAM/work_dir/MedSAM/medsam_vit_b.pth
```

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv medsam_env
source medsam_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## MedSAM Setup

This repository includes the MedSAM source code, but not the model checkpoint.

Download the MedSAM ViT-B checkpoint separately and place it here:

```text
MedSAM/work_dir/MedSAM/medsam_vit_b.pth
```

The evaluation script expects this path by default.

---

## Dataset Setup

The CAMRI MRI data and expert masks should be stored locally using the following structure:

```text
Image Database/
└── CAMRI Rat Brain MRI Data/

Mask_Database/
└── RodentBrainMask/
    └── CAMRI Rat/
```

These folders are excluded from GitHub because they contain large imaging data.

---

## Running the Main Benchmark

Run the full CAMRI rat MedSAM benchmark:

```bash
MPLBACKEND=Agg medsam_env/bin/python scripts/evaluate_medsam_camri_rat.py
```

Run a smoke test on one subject:

```bash
MPLBACKEND=Agg medsam_env/bin/python scripts/evaluate_medsam_camri_rat.py --max-subjects 1
```

The script produces:

```text
medsam_camri_rat_results.csv
outputs/camri_rat_examples/
```

---

## Generating Result Figures

After running the benchmark, generate summary figures and reports:

```bash
MPLBACKEND=Agg medsam_env/bin/python scripts/create_medsam_results_figures.py
```

This creates:

```text
outputs/results_figures/
```

including summary tables, Dice distributions, per-subject plots, and worst-case slice analysis.

---

## WhiteStripe Diagnostic Script

`whitestripe_mouse_diagnostic.py` is an exploratory preprocessing utility. It visualizes intensity histograms and WhiteStripe-style normalization for mouse MRI data.

It is not required for the main CAMRI MedSAM benchmark.

---

## Notes for Collaborators

- Do not commit raw medical imaging data.
- Do not commit model checkpoints.
- Do not commit the virtual environment.
- Commit source code, selected result images, CSV summaries, and documentation.
- If file paths change, update the default paths inside the scripts or pass paths through command-line arguments.

---

## Author

Esteban Felix  
Computer Engineering  
Purdue University Indianapolis