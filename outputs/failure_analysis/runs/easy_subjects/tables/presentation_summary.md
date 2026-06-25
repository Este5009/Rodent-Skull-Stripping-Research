# MedSAM Failure Analysis Summary

## Overall Run Summary
| Item | Value |
|---|---|
| Selected Subjects | sub-001, sub-002, sub-003, sub-004, sub-005 |
| Number Of Subjects | 5 |
| Number Of Slices | 60 |
| Mean Dice | 0.9652 |
| Median Dice | 0.9685 |
| Worst Dice | 0.9350 |
| Best Dice | 0.9801 |

## Failure Interpretation
| Item | Value |
|---|---|
| Typical Performance | median Dice 0.9685 |
| Outlier Effect | mean Dice is lower than median Dice by 0.0033 |
| Worst Failure | sub-004, slice 1, Dice 0.9350 |
| Main Suspected Failure Type | over-segmentation / wrong structure |
| Small Slice Evidence | worst 10 mean mask area 12038.4 px; all-slice mean 11136.1 px |

