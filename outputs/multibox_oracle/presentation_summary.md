# MedSAM Multi-Box Oracle: Presentation Summary

## Question

Can adaptive non-overlapping component prompts and optional local clipping
improve on the previous fixed-margin multi-box strategy?

## Run

- Subjects: sub-086
- Non-empty slices: 55
- Box margin: 5 pixels
- Minimum retained component area: 10 pixels
- Merge distance threshold: 3.0 pixels
- Local clipping enabled: True
- Slices with multiple GT components: 14
- Slices receiving multiple MedSAM boxes: 12
- Slices with overlapping raw boxes: 9
- Slices with merged adaptive prompts: 6

## Result

- Mean single-box Dice: 0.8716
- Mean raw multi-box Dice: 0.9172
- Mean adaptive Dice: 0.9185
- Mean raw improvement over single: +0.0455
- Mean adaptive improvement over raw: +0.0013
- Mean adaptive + clipping Dice: 0.9187
- Mean Dice change on multi-component GT slices: +0.1789
- Mean Dice change on slices receiving multiple boxes: +0.1312
- Adaptive improved / worsened / unchanged slices: 4 / 5 / 46
- Clipping improved / worsened / unchanged slices: 1 / 0 / 54

- sub-086: 0.8716 -> 0.9172 -> 0.9185 -> 0.9187 (clipped)

## Interpretation

Adaptive prompts improved over raw multi-box prompting in this run. This is an oracle prompt-geometry diagnostic, not evidence that an
automatic proposal network can yet reproduce the oracle components.
