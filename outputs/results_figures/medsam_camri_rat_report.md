# MedSAM CAMRI Rat Oracle-Box Results

## Dataset Summary

- Subjects evaluated: **131**
- Slices evaluated: **4563**

## Metric Summary

| Metric | Mean +/- SD | Median | Min | Max |
|---|---:|---:|---:|---:|
| Dice | 0.9111 +/- 0.1470 | 0.9621 | 0.0000 | 0.9909 |
| IoU | 0.8604 +/- 0.1781 | 0.9269 | 0.0000 | 0.9820 |
| Precision | 0.9065 +/- 0.1893 | 0.9914 | 0.0000 | 1.0000 |
| Recall | 0.9433 +/- 0.0542 | 0.9526 | 0.0000 | 1.0000 |
| Hausdorff (px) | 9.2831 +/- 6.6183 | 7.2111 | 1.0000 | 43.0465 |

## Best and Worst Cases

- Best slice: **sub-075**, slice **31**, Dice **0.9909**
- Worst slice: **sub-086**, slice **55**, Dice **0.0000**
- Best subject by mean Dice: **sub-033**, mean Dice **0.9737**
- Worst subject by mean Dice: **sub-112**, mean Dice **0.8645**

## Worst 20 Slices Ranked by Dice

| Rank | Subject | Slice | Dice | IoU | Precision | Recall | Hausdorff px |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | sub-086 | 55 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 43.05 |
| 2 | sub-119 | 56 | 0.0068 | 0.0034 | 0.0037 | 0.0448 | 40.00 |
| 3 | sub-106 | 58 | 0.0207 | 0.0104 | 0.0108 | 0.2473 | 40.16 |
| 4 | sub-086 | 54 | 0.0249 | 0.0126 | 0.0142 | 0.1022 | 39.62 |
| 5 | sub-087 | 55 | 0.0254 | 0.0128 | 0.0131 | 0.4058 | 39.12 |
| 6 | sub-112 | 56 | 0.0278 | 0.0141 | 0.0148 | 0.2222 | 42.43 |
| 7 | sub-066 | 59 | 0.0370 | 0.0188 | 0.0191 | 0.5821 | 35.47 |
| 8 | sub-084 | 50 | 0.0444 | 0.0227 | 0.0227 | 1.0000 | 6.40 |
| 9 | sub-108 | 62 | 0.0476 | 0.0244 | 0.0244 | 1.0000 | 4.47 |
| 10 | sub-062 | 59 | 0.0741 | 0.0385 | 0.0385 | 1.0000 | 5.83 |
| 11 | sub-046 | 54 | 0.0769 | 0.0400 | 0.0400 | 1.0000 | 5.39 |
| 12 | sub-076 | 6 | 0.0774 | 0.0403 | 0.0403 | 1.0000 | 16.55 |
| 13 | sub-050 | 4 | 0.0816 | 0.0426 | 0.0426 | 1.0000 | 4.47 |
| 14 | sub-065 | 55 | 0.0901 | 0.0472 | 0.0476 | 0.8364 | 32.25 |
| 15 | sub-095 | 54 | 0.0957 | 0.0503 | 0.0512 | 0.7391 | 39.05 |
| 16 | sub-060 | 3 | 0.0981 | 0.0516 | 0.0588 | 0.2955 | 15.30 |
| 17 | sub-082 | 57 | 0.1004 | 0.0529 | 0.0547 | 0.6157 | 34.83 |
| 18 | sub-070 | 53 | 0.1036 | 0.0546 | 0.0592 | 0.4145 | 31.02 |
| 19 | sub-091 | 55 | 0.1064 | 0.0562 | 0.0608 | 0.4263 | 33.53 |
| 20 | sub-112 | 61 | 0.1081 | 0.0571 | 0.0571 | 1.0000 | 5.00 |

## Generated Figures

- `summary_statistics_table.png`
- `bar_mean_overlap_metrics.png`
- `boxplots_dice_iou_hausdorff.png`
- `histogram_dice_scores.png`
- `scatter_gt_area_vs_dice.png`
- `per_subject_dice_distribution.png`
- `worst_20_slices_by_dice.png`

All figures are also saved as PDF files for manuscript workflows.
