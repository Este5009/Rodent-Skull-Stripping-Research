# AGENTS.md

## Read First

Before making changes, read:

```text
AGENTS.md
MEMORY.md
```

Use AGENTS.md for project goals and repository rules.
Use MEMORY.md for user preferences.

If you discover a durable preference about coding style, documentation style, repository organization, or explanation style, suggest adding it to MEMORY.md.

---

## Research Vision

Goal:

```text
Build a highly accurate, autonomous, and generalizable medical / biomedical image segmentation system.
```

The current rodent MRI skull-stripping task is only the first benchmark used to validate ideas and measure progress toward that larger goal.

Do not frame the project as:

* only a rodent brain segmentation project
* only a MedSAM project
* only a prompt-generation project

MedSAM is currently one candidate segmentation backend inside a larger autonomous segmentation pipeline.

---

## Current Research Question

Current benchmark:

```text
CAMRI rat MRI + expert masks
```

Current observation:

```text
MedSAM + oracle boxes ≈ 0.91 Dice
```

Main question:

```text
Why is oracle-box MedSAM underperforming compared with rodent-specific supervised models?
```

Before proposing U-Net, YOLO, or new architectures, first investigate:

* box-margin sensitivity
* preprocessing sensitivity
* implementation issues
* domain-gap limitations

---

## Repository Rules

* `scripts/core/` = validated/stable scripts
* `scripts/experimental/` = new experiments
* do not modify core scripts unless explicitly requested
* reuse existing code whenever possible
* preserve working code
* prefer incremental improvements over rewrites

Reference implementation:

```text
scripts/core/evaluate_medsam_camri_rat.py
```

---

## Current Phase

Phase 1:

```text
Understand MedSAM behavior.
```

Priority experiments:

1. smoke tests
2. box-margin sensitivity
3. preprocessing sensitivity
4. metric analysis
5. visual QC

Only after Phase 1 should automatic box generation or U-Net comparisons become priorities.

---

## Safety

This repository is a research prototype, not clinical software.
