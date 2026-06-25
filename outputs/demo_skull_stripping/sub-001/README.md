# MedSAM Skull-Stripping Demo: sub-001

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
