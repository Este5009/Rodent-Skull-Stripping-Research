import argparse

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Default DICOM used when the script is run without arguments:
#   medsam_env/bin/python whitestripe_mouse_diagnostic.py
# Use --dicom to inspect a different acquisition without editing the script.
DEFAULT_DICOM = (
    "2024-12__Studies/"
    "Johnson.JJ3.302_JJ3.302_MR_2024-12-03_181547_._no.gap.T2.TurboRARE_n13__00000/"
    "2.16.756.5.5.200.8323328.42181.1733268218.4802.3.0.dcm"
)


def parse_box(value):
    """Read an ROI box written as x_min,y_min,x_max,y_max."""
    # Command-line arguments arrive as strings, so split on commas and convert
    # before doing any geometric validation.
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x_min,y_min,x_max,y_max")
    x_min, y_min, x_max, y_max = parts
    # Require a positive-width, positive-height ROI; otherwise the histogram
    # would be empty or inverted.
    if x_min >= x_max or y_min >= y_max:
        raise argparse.ArgumentTypeError("box max values must be greater than min values")
    return x_min, y_min, x_max, y_max


def robust_clip(values, low=0.5, high=99.5):
    """Remove extreme low/high values before building the histogram.

    MRI DICOMs often contain bright scanner/artifact tails. Clipping those tails
    makes tissue peaks visible without changing the raw values used later for
    the selected stripe.
    """
    # Percentiles are estimated from the ROI values only, so clipping reflects
    # the tissue/background mixture inside the selected diagnostic region.
    lower, upper = np.percentile(values, [low, high])
    return values[(values >= lower) & (values <= upper)], lower, upper


def find_histogram_peaks(values, bins=256, sigma=2.0):
    """Build a smoothed histogram and rank local maxima by peak height."""
    # The histogram is built in raw intensity space so the reported peaks can be
    # reused as WhiteStripe anchors in other scripts.
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    # Smoothing suppresses bin-to-bin noise without fitting a full mixture model.
    smooth = gaussian_filter1d(counts.astype(np.float32), sigma=sigma)
    peaks, props = find_peaks(
        smooth,
        prominence=max(float(smooth.max()) * 0.03, 1.0),
        distance=max(3, bins // 40),
    )
    # Rank by smoothed count so peak index 0 is the dominant histogram peak, not
    # necessarily the anatomically correct tissue peak.
    order = np.argsort(smooth[peaks])[::-1]
    peaks = peaks[order]
    return centers, counts, smooth, peaks, props


def whitestripe_normalize(image, stripe_values):
    """Z-score an image using voxels from the selected WhiteStripe band."""
    # WhiteStripe normalization assumes the chosen band is a stable tissue class.
    # If the band is too narrow or empty, the standard deviation is not usable.
    mu = float(np.mean(stripe_values))
    sigma = float(np.std(stripe_values))
    if sigma < 1e-6:
        raise ValueError("WhiteStripe standard deviation is too small")
    return (image.astype(np.float32) - mu) / sigma, mu, sigma


def to_uint8_for_medsam(image, low=0.5, high=99.5):
    """Convert a 2D image to an 8-bit display image for visual inspection."""
    # Use a robust display window so one extreme voxel does not make the preview
    # look uniformly dark.
    lo, hi = np.percentile(image, [low, high])
    clipped = np.clip(image, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6)
    return (scaled * 255).astype(np.uint8)


def main():
    # The command-line knobs are intentionally lightweight; this script is meant
    # for quick histogram checks while deciding which peak is anatomically useful.
    parser = argparse.ArgumentParser(
        description="Plot mouse MRI ROI histogram and WhiteStripe-style normalization."
    )
    parser.add_argument("--dicom", default=DEFAULT_DICOM)
    parser.add_argument("--slice", type=int, default=6)
    parser.add_argument("--box", type=parse_box, default=parse_box("65,95,195,190"))
    parser.add_argument("--peak-index", type=int, default=0)
    parser.add_argument("--stripe-width", type=float, default=0.05)
    parser.add_argument("--output", default="whitestripe_diagnostic.png")
    args = parser.parse_args()

    # The DICOM used in the mouse experiments stores the image stack in
    # pixel_array with shape (slice, row, col).
    dcm = pydicom.dcmread(args.dicom)
    volume = dcm.pixel_array.astype(np.float32)
    if not 0 <= args.slice < volume.shape[0]:
        raise ValueError(f"--slice must be in [0, {volume.shape[0] - 1}]")

    # Estimate the histogram from an ROI rather than the whole image. Background
    # and skull can dominate a rodent MRI histogram and hide the tissue peaks we
    # care about for normalization.
    x_min, y_min, x_max, y_max = args.box
    # In image coordinates x indexes columns and y indexes rows. The ROI is
    # applied to every slice to collect enough voxels for a stable histogram.
    roi = volume[:, y_min:y_max, x_min:x_max].reshape(-1)
    roi = roi[np.isfinite(roi)]
    clipped_roi, clip_low, clip_high = robust_clip(roi)

    # Peak ranking is by smoothed histogram height, not by anatomical class. The
    # highest peak can still be background, so the printed candidates need visual
    # interpretation.
    centers, counts, smooth, peaks, _ = find_histogram_peaks(clipped_roi)
    if len(peaks) == 0:
        raise RuntimeError("No histogram peaks found. Try a tighter brain ROI box.")
    if not 0 <= args.peak_index < len(peaks):
        raise ValueError(f"--peak-index must be in [0, {len(peaks) - 1}]")

    peak_center = float(centers[peaks[args.peak_index]])
    # Use a relative stripe width so the band scales with raw intensity. This is
    # simple and adequate for exploratory peak testing.
    half_width = peak_center * args.stripe_width

    # The WhiteStripe band is centered on the selected raw-intensity peak. Its
    # mean/std define the z-score reference for the whole volume.
    stripe_mask = (roi >= peak_center - half_width) & (roi <= peak_center + half_width)
    stripe_values = roi[stripe_mask]
    if stripe_values.size < 20:
        raise RuntimeError("Too few voxels in stripe. Increase --stripe-width.")

    # Normalize the whole volume from the selected stripe, then display only the
    # requested slice so the preview matches the slice used for MedSAM testing.
    norm_volume, stripe_mu, stripe_sigma = whitestripe_normalize(volume, stripe_values)
    medsam_slice = to_uint8_for_medsam(norm_volume[args.slice])

    print(f"Volume shape: {volume.shape}")
    print(f"ROI box: {args.box}; ROI robust clip: [{clip_low:.2f}, {clip_high:.2f}]")
    print("Candidate peaks, ranked by histogram height:")
    for rank, idx in enumerate(peaks[:8]):
        print(f"  {rank}: intensity={centers[idx]:.2f}, smoothed_count={smooth[idx]:.0f}")
    print(f"Selected peak {args.peak_index}: {peak_center:.2f}")
    print(f"WhiteStripe mean={stripe_mu:.2f}, std={stripe_sigma:.2f}, voxels={stripe_values.size}")
    print(f"Saved: {args.output}")

    # Plot the ROI, histogram, and normalized preview together so peak choice can
    # be judged against anatomy rather than the histogram alone.
    slice_raw = volume[args.slice]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].imshow(to_uint8_for_medsam(slice_raw), cmap="gray")
    # Draw the same ROI used for histogram estimation, not the MedSAM prompt box.
    axes[0].add_patch(
        plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="lime",
            linewidth=1.5,
        )
    )
    axes[0].set_title(f"Slice {args.slice} with ROI box")
    axes[0].axis("off")

    # Plot raw and smoothed histograms together so the peak detector's decisions
    # can be compared against the unsmoothed data.
    axes[1].plot(centers, counts, color="0.7", linewidth=1, label="raw hist")
    axes[1].plot(centers, smooth, color="black", linewidth=1.5, label="smoothed")
    axes[1].axvspan(
        peak_center - half_width,
        peak_center + half_width,
        color="tab:orange",
        alpha=0.25,
        label="stripe",
    )
    for rank, idx in enumerate(peaks[:8]):
        # Label candidate peaks by rank; use --peak-index with these numbers.
        axes[1].axvline(centers[idx], linestyle="--", linewidth=0.9)
        axes[1].text(centers[idx], smooth[idx], str(rank), fontsize=8)
    axes[1].set_title("ROI intensity histogram")
    axes[1].set_xlabel("Raw DICOM intensity")
    axes[1].set_ylabel("Voxel count")
    axes[1].legend(fontsize=8)

    # The preview is not itself a segmentation input unless the same peak/window
    # choice is copied into a MedSAM inference script.
    axes[2].imshow(medsam_slice, cmap="gray")
    axes[2].set_title("WhiteStripe-normalized preview")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
