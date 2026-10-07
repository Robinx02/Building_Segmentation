# Building Footprint Segmentation

Semantic segmentation of building footprints from high-resolution satellite imagery using a lightweight U-Net with boundary-aware post-processing, benchmarked against a classical threshold baseline.

> **Course:** Computer Vision Project  
> **Dataset:** [WHU Building Dataset — Satellite Dataset I](https://gpcv.whu.edu.cn/data/building_dataset.html)  
> **Framework:** PyTorch

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)

---

## Overview

Given a 512×512 satellite image tile, the model predicts a binary mask where:

- **White (255)** = building pixel
- **Black (0)** = background pixel

We compare three approaches:

| Approach | Description |
|---|---|
| Classical Baseline | Otsu thresholding + morphological cleanup |
| U-Net (BCE + Dice) | Lightweight encoder-decoder with skip connections |
| U-Net + Boundary Loss | Same model with boundary-weighted loss / post-processing |

---

## Dataset

**WHU Building Dataset — Satellite Dataset I**

| Property | Value |
|---|---|
| Total images | 204 tiles |
| Image size | 512 × 512 pixels |
| Resolution | 0.3 m – 2.5 m per pixel |
| Annotation | Binary PNG mask per tile |
| Sensor sources | QuickBird, Worldview, IKONOS, ZY-3 |
| Download size | ~113 MB |

Download the dataset from the [official GPCV page](https://gpcv.whu.edu.cn/data/building_dataset.html) and place it at:

```
data/raw/satellite_dataset_I/
    image/      ← 204 RGB tiles (.png)
    label/      ← 204 binary masks (.png)
```

> The `data/raw/` folder is gitignored — do not commit the dataset.

---

## Project Structure

```
building-segmentation/
│
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── task.md
│
├── data/
│   ├── raw/                        # ← gitignored (put downloaded WHU zip here)
│   │   └── satellite_dataset_I/
│   │       ├── image/              # 204 RGB tiles (.png)
│   │       └── label/              # 204 binary masks (.png)
│   │
│   ├── processed/                  # ← gitignored (normalized/resized tiles)
│   │   ├── images/
│   │   └── masks/
│   │
│   └── splits/                     # ← committed to git
│       ├── train.txt               # filenames for training  (~160 images)
│       ├── val.txt                 # filenames for validation (~24 images)
│       └── test.txt                # filenames for testing   (~20 images)
│
├── notebooks/
│   ├── 01_eda.ipynb                # dataset exploration, class distribution
│   ├── 02_baseline.ipynb           # Otsu + morphology baseline
│   └── 03_results.ipynb            # final results, failure analysis
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # WHUDataset class, augmentations
│   ├── model.py                    # U-Net architecture
│   ├── loss.py                     # Dice, BCE+Dice, BoundaryWeightedLoss
│   ├── train.py                    # training + validation loop
│   ├── evaluate.py                 # IoU, Dice, Precision, Recall, Boundary F1
│   ├── baseline.py                 # Otsu threshold + morphological ops
│   └── postprocess.py              # boundary refinement post-processing
│
├── configs/
│   ├── baseline.yaml               # baseline experiment config
│   ├── unet_bce_dice.yaml          # standard U-Net config
│   └── unet_boundary.yaml          # boundary-aware U-Net config
│
├── outputs/                        # ← gitignored (generated during runs)
│   ├── checkpoints/                # saved model weights (.pth)
│   ├── predictions/                # output masks (baseline/ and unet/)
│   ├── plots/                      # training curves, failure analysis images
│   └── metrics.csv                 # final IoU/Dice/F1 results table
│
├── report/
│   ├── report.pdf                  # final written report
│   └── figures/                    # diagrams for report and PPT
│       ├── unet_architecture.png
│       ├── pipeline.png
│       └── sample_predictions.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

---
