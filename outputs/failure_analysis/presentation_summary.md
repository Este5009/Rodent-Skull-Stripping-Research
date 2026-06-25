# MedSAM Failure Analysis Summary

## Overall Run Summary
| Item | Value |
|---|---|
| Selected Subjects | sub-050, sub-066, sub-086, sub-109, sub-112 |
| Number Of Subjects | 5 |
| Number Of Slices | 276 |
| Mean Dice | 0.8682 |
| Median Dice | 0.9531 |
| Worst Dice | 0.0000 |
| Best Dice | 0.9889 |

## Easy vs Hard Comparison
| Group | Subjects | Slices | Mean Dice | Median Dice | Worst Dice | Precision | Recall | Mask Area | Box Area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 0 | 0 | not available | not available | not available | not available | not available | not available | not available |
| Hard / current | 5 | 276 | 0.8682 | 0.9531 | 0.0000 | 0.8565 | 0.9280 | 4555.2 | 8453.7 |

## Failure Interpretation
| Item | Value |
|---|---|
| Typical Performance | median Dice 0.9531 |
| Outlier Effect | mean Dice is lower than median Dice by 0.0849 |
| Worst Failure | sub-086, slice 55, Dice 0.0000 |
| Main Suspected Failure Type | complete failure |
| Small Slice Evidence | worst 10 mean mask area 183.7 px; all-slice mean 4555.2 px |

