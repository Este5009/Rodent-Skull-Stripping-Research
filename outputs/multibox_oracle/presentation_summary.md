# MedSAM Multi-Box Oracle: Presentation Summary

## Question

Does replacing one whole-mask oracle box with independent 8-connected component
boxes improve MedSAM segmentation on difficult CAMRI slices?

## Run

- Subjects: sub-086
- Non-empty slices: 55
- Box margin: 5 pixels
- Minimum retained component area: 10 pixels
- Slices with multiple GT components: 14
- Slices receiving multiple MedSAM boxes: 12

## Result

- Mean single-box Dice: 0.8716
- Mean multi-box Dice: 0.9172
- Mean Dice change: +0.0455
- Mean Dice change on multi-component GT slices: +0.1789
- Mean Dice change on slices receiving multiple boxes: +0.1312
- Improved / worsened / unchanged slices: 11 / 2 / 42

- sub-086: 0.8716 -> 0.9172 (+0.0455)

## Interpretation

Multi-box oracle prompts improved mean Dice, supporting prompt geometry as a contributor. This is an oracle prompt-geometry diagnostic, not evidence that an
automatic proposal network can yet reproduce the oracle components.
