import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. Paths
# ----------------------------
# Fixed subject used for a quick visual sanity check. This script is intentionally
# minimal; the full quantitative benchmark lives in evaluate_medsam_camri_rat.py.
# Keeping the paths explicit makes it easy to swap in a suspect subject when
# checking alignment or orientation issues by hand.
mri_path = (
    "Image Database/CAMRI Rat Brain MRI Data/"
    "sub-001/ses-1/anat/sub-001_ses-1_acq-RARE_T2w.nii.gz"
)

mask_path = (
    "Mask_Database/RodentBrainMask/CAMRI Rat/"
    "CAMRI_Rat-sub-001_ses-1_acq-RARE_T2w_061.nii.gz"
)

# ----------------------------
# 2. Load MRI and mask
# ----------------------------
# get_fdata() applies NIfTI scaling and returns floating point arrays. That is
# useful for display, but evaluation code keeps stricter control over dtypes.
mri = nib.load(mri_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# Print basic metadata before plotting so shape/value problems are visible even
# when running in a non-interactive backend.
print("MRI shape:", mri.shape)
print("Mask shape:", mask.shape)
print("MRI min/max:", np.min(mri), np.max(mri))
print("Mask unique values:", np.unique(mask))

# ----------------------------
# 3. Pick middle slice
# ----------------------------
# CAMRI volumes are stored as (x, y, z), so axis 2 indexes the slice direction.
# The middle slice is usually a good first check because the brain mask is large
# enough there to reveal registration or orientation errors.
mid = mri.shape[2] // 2

mri_slice = mri[:, :, mid]
mask_slice = mask[:, :, mid]

# ----------------------------
# 4. Show all three together
# ----------------------------
# This view checks that the MRI and manual mask occupy the same voxel grid before
# running any automated MedSAM inference or metric calculation.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 is the raw image slice, without normalization beyond Matplotlib's
# display scaling.
axes[0].imshow(mri_slice, cmap="gray")
axes[0].set_title("Original MRI")
axes[0].axis("off")

# Panel 2 shows the mask alone so non-binary labels or holes are easy to spot.
axes[1].imshow(mask_slice, cmap="gray")
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

# Panel 3 overlays the mask on the image; this is the fastest check that image
# and annotation are aligned voxel-for-voxel.
axes[2].imshow(mri_slice, cmap="gray")
axes[2].imshow(mask_slice, cmap="jet", alpha=0.4)
axes[2].set_title("Mask Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
