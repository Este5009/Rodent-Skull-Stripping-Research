
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Quick DICOM slice viewer used while hand-tuning early MedSAM box prompts. The
# oracle-box benchmark now creates boxes from expert masks automatically.
dcm = pydicom.dcmread("2024-12__Studies/Johnson.JJ3.302_JJ3.302_MR_2024-12-03_181547_._no.gap.T2.TurboRARE_n13__00000/2.16.756.5.5.200.8323328.42181.1733268218.4802.3.0.dcm")

# For this multi-frame DICOM, pixel_array is indexed as (slice, row, column).
volume = dcm.pixel_array
print("Volume shape:", volume.shape)

# Use a representative middle-ish slice for prompt debugging.
slice_index = 6
img = volume[slice_index]

# Box format follows image coordinates: x is column, y is row. These values are
# only for visual prompt debugging; they are not ground-truth annotations.
box = [65, 95, 195, 190]   # adjust these numbers after seeing the image

# Plot the image first, then draw the rectangle in the same pixel coordinate
# system that MedSAM expects for box prompts.
plt.imshow(img, cmap="gray")

x_min, y_min, x_max, y_max = box
# Rectangle wants the top-left corner plus width/height, while MedSAM boxes are
# stored as two corners. Convert here only for visualization.
rect = Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    fill=False,
    linewidth=2
)

plt.gca().add_patch(rect)
plt.title(f"Slice {slice_index}")
plt.axis("off")
plt.show()
