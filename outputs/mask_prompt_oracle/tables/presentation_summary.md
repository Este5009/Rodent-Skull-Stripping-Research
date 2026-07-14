# Coarse Mask-Prompt Oracle Summary

## Question

Can a deliberately coarse, shape-aware mask prompt solve difficult MedSAM
failures better than oracle rectangular prompts?

## Run

- Subjects: sub-086
- Non-empty slices: 55
- Coarse bottleneck: 32 x 32
- Perturbation: none, radius 1

## Results

- Single oracle box: mean Dice 0.8716
- Raw multi-box: mean Dice 0.9172
- Coarse mask only: mean Dice 0.0000
- Multi-box + coarse mask: mean Dice 0.9251

- Best mean method: Multi-box + coarse mask (0.9251)
- Mean best-mask improvement over best-box: +0.0078
- Improved / worsened / unchanged slices: 47 / 8 / 0

## Interpretation

This is an oracle feasibility result. The prompt has passed through a
low-resolution bottleneck and is not an exact expert boundary, but an automatic
proposal network must still learn to generate comparable coarse shapes.
