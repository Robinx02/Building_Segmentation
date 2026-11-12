# Building Footprint Segmentation

Semantic segmentation of building footprints from high-resolution satellite imagery. A U-Net (trained with BCE + Dice, and with an edge-weighted variant) is benchmarked against classical machine-learning and deep-learning baselines on the WHU Building Dataset, using one leakage-free split and one shared set of metrics.

> **Course:** Computer Vision Project
> **Dataset:** [WHU Building Dataset, Satellite Dataset I](https://gpcv.whu.edu.cn/data/building_dataset.html)
> **Framework:** PyTorch

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Method](#method)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Reproducing the Results](#reproducing-the-results)
- [Findings](#findings)
- [Limitations and Notes](#limitations-and-notes)
- [Team](#team)
- [Citation](#citation)

---

## Overview

Given a 512×512 satellite image tile, the model predicts a binary mask where:

- **White (255)** = building pixel
- **Black (0)** = background pixel

Models compared (all trained from scratch, no pretrained weights):

| Approach | Description |
|---|---|
| Decision Tree | Per-pixel classifier on R, G, B and LBP texture features |
| Random Forest | 50-tree ensemble on the same per-pixel features |
| DeepLabV3 (ResNet-50) | Atrous spatial pyramid pooling, 1-channel output |
| FPN (ResNet-18) | Feature pyramid network from `segmentation_models_pytorch` |
| **U-Net (BCE + Dice)** | 31.04 M-parameter encoder-decoder with skip connections |
| **U-Net + boundary-weighted loss** | Same model, BCE and Dice terms weighted towards building edges |

---

## Results

Test split (20 tiles). Every model uses the same train/val/test split and the same metric code. Boundary F1 uses a 3 px tolerance.

| Method | IoU | Dice | Precision | Recall | Boundary F1 |
|---|---|---|---|---|---|
| **U-Net (BCE + Dice)** | **0.6319** | **0.7704** | **0.7706** | 0.7761 | 0.6024 |
| U-Net (boundary-weighted) | 0.6287 | 0.7682 | 0.7545 | **0.7885** | **0.6199** |
| Decision Tree | 0.4019 | 0.5635 | 0.4910 | 0.6827 | 0.3669 |
| Random Forest | 0.3973 | 0.5577 | 0.4593 | 0.7481 | 0.3674 |
| DeepLabV3 | 0.3866 | 0.5460 | 0.7672 | 0.4439 | 0.3156 |
| FPN (ResNet-18) | 0.3223 | 0.4702 | 0.7482 | 0.3705 | 0.2755 |

**Boundary-aware extension (U-Net, test split):** the edge-weighted loss raises Boundary F1 from 0.6024 to 0.6199 (+0.0175). Boundary recall rises from 0.626 to 0.718 and boundary precision falls from 0.584 to 0.550. IoU is essentially unchanged (0.6319 to 0.6287). One seed and 20 test tiles, so treat the gain as suggestive.

**Training budgets differ.** The baselines were trained for 10 epochs with plain BCE and no augmentation; the U-Net for up to 80 epochs with BCE + Dice, augmentation and validation-based model selection. Part of the gap is therefore training budget, not architecture alone.

---

## Dataset

**WHU Building Dataset, Satellite Dataset I**

| Property | Value |
|---|---|
| Total images | 204 tiles |
| Image size | 512 × 512 pixels |
| Resolution | 0.3 m to 2.5 m per pixel |
| Annotation | Binary mask per tile |
| Sensor sources | QuickBird, Worldview, IKONOS, ZY-3 |
| Download size | ~113 MB |

Download the dataset from the [official GPCV page](https://gpcv.whu.edu.cn/data/building_dataset.html). Each training pipeline expects a folder that contains `image/` and `label/` with matching file names:

```
<data_root>/
    image/      <- 204 RGB tiles
    label/      <- 204 binary masks
```

The U-Net scripts take `--data_root` (default `data/raw`). The baselines read `baseline_nbs/data/raw/satellite_dataset_I/` (see `baseline_nbs/src/config.py`).

> `data/raw/` is gitignored. Do not commit the dataset.

---

## Method

### Leakage-free split

`src/split_data.py` finds neighbouring tiles by comparing the last row/column of one tile with the first row/column of another (border difference relative to the typical in-tile difference, ratio below 1.5, mutual best match). Linked tiles form groups that are never divided. Groups are placed largest first into the split furthest below its target of **160 train / 24 validation / 20 test** (seed 42). The script asserts that splits are disjoint, nothing is lost and no group straddles two splits. Tiles that overlap (rather than touch) are not detected.

Normalisation statistics come from the training tiles only, and cropping and augmentation happen after the split. The baselines are trained and tested on this same split: copy `data/splits/{train,val,test}.txt` into `baseline_nbs/data/splits/`.

### U-Net

- Standard 4-level U-Net: 64, 128, 256, 512 channels, 1024-channel bottleneck, transposed-convolution decoder, skip connections by concatenation, 1×1 output convolution (31.04 M parameters).
- Trained on random 256×256 crops (each tile used 4× per epoch); evaluated on full 512×512 tiles.
- Augmentation (train only): flips, 90° rotations, small rotations (±15°, p = 0.3), brightness/contrast jitter.
- AdamW (lr 1e-3, weight decay 1e-4), batch size 8, learning rate halved when validation IoU stalls for 5 epochs, mixed precision on CUDA.
- Up to 80 epochs with early stopping (patience 20); the checkpoint with the best validation IoU is evaluated once on the test split. Seed 42.

### Losses (`src/loss.py`)

| `--loss` | Definition |
|---|---|
| `bce_dice` | 0.5·BCE + 0.5·Dice |
| `bce_iou` | 0.5·BCE + 0.5·IoU loss |
| `boundary` | BCE and Dice weighted by `1 + edge_weight · band`, where `band` marks pixels within `edge_width` px of a building edge (defaults: weight 5, width 2) |

### Baselines (`baseline_nbs/`)

- **Decision Tree / Random Forest:** per-pixel features B, G, R and uniform LBP (radius 1, 8 points); up to 2,500 background and 2,500 building pixels sampled per training image. Decision Tree `max_depth=15`; Random Forest `n_estimators=50`, `max_depth=10`.
- **DeepLabV3 (ResNet-50)** and **FPN (ResNet-18):** trained from scratch (`weights=None`, `weights_backbone=None` / `encoder_weights=None`) with BCE, Adam (lr 1e-4), batch size 2, 10 epochs, full 512×512 tiles, no augmentation.

### Metrics (`src/metrics.py`)

IoU, Dice/F1, precision and recall are computed per image and averaged. **Boundary F1** uses the one-pixel inner boundary of each mask (mask minus its 3×3 erosion): a boundary pixel counts as matched if the other mask's boundary lies within 3 px (distance transform). Boundary precision and recall are combined into F1.

---

## Project Structure

```
Building_segmentation/
├── baseline_nbs/                   # classical + deep baselines
│   ├── main.py                     # --model {decision_tree,random_forest,resnet18,deeplabv3} --mode {train,test}
│   ├── split_data.py               # sorted-name split (superseded by src/split_data.py)
│   ├── requirements.txt
│   └── src/
│       ├── config.py               # paths, epochs, batch size, LBP settings
│       ├── dataset.py              # WHUDataset, LBP feature extraction
│       ├── engine.py               # train / test loops for all baselines
│       ├── evaluate.py             # original baseline metrics (superseded by src/metrics.py)
│       └── models/
│           ├── traditional.py      # Decision Tree, Random Forest
│           └── deep.py             # DeepLabV3, FPN-ResNet-18
│
├── src/                            # U-Net pipeline
│   ├── split_data.py               # adjacency-grouped, leakage-free split
│   ├── data.py                     # dataset, augmentation, train-only normalisation
│   ├── unet_arch.py                # U-Net
│   ├── loss.py                     # BCE+Dice, BCE+IoU, boundary-weighted loss
│   ├── metrics.py                  # IoU, Dice, precision, recall, Boundary F1
│   ├── train.py                    # training, validation, final test evaluation
│   └── rescore_baselines.py        # score baseline predictions with src/metrics.py
│
├── data/
│   ├── raw/                        # gitignored (WHU download)
│   └── splits/                     # train.txt, val.txt, test.txt, split_manifest.csv
├── configs/
├── outputs/                        # gitignored
│   ├── checkpoints/                # best-on-validation weights
│   ├── history/                    # per-epoch training logs
│   ├── per_image/                  # per-image metrics
│   ├── plots/                      # training curves
│   ├── predictions/                # predicted masks
│   └── metrics.csv                 # results table (one row per run and split)
├── report/                         # written report and figures
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

Developed with Python 3.13 and a CUDA GPU (CPU also works, but slowly).

```bash
pip install -r requirements.txt
pip install -r baseline_nbs/requirements.txt   # scikit-image, scikit-learn, segmentation-models-pytorch, joblib
```

---

## Reproducing the Results

Run everything from the repository root unless noted. The U-Net scripts use paths relative to the working directory.

**1. Create the split**

```bash
python src/split_data.py --root data/raw --out_dir data/splits
```

**2. Train and evaluate the U-Net**

```bash
python src/train.py --loss bce_dice      # plain U-Net
python src/train.py --loss boundary      # boundary-weighted loss
```

Each run appends rows (val and test) to `outputs/metrics.csv`, and writes predictions, per-image metrics, training history and curves under `outputs/`.

**3. Train and test the baselines on the same split**

```bash
cp data/splits/*.txt baseline_nbs/data/splits/
cd baseline_nbs
for m in decision_tree random_forest resnet18 deeplabv3; do
  python main.py --model $m --mode train --split train.txt
  python main.py --model $m --mode test  --split test.txt
done
cd ..
```

On Windows PowerShell:

```powershell
foreach ($m in "decision_tree","random_forest","resnet18","deeplabv3") {
  python main.py --model $m --mode train --split train.txt
  python main.py --model $m --mode test --split test.txt
}
```

**4. Score the baselines with the shared metrics**

```bash
python src/rescore_baselines.py
```

Check the `LABELS` and `PREDS` paths at the top of the script before running. `main.py --mode test` also appends rows to `baseline_nbs/outputs/metrics.csv`, but those use the older Boundary F1 definition; use the rescored numbers.

---

## Findings

The U-Net's remaining errors on the test tiles fall into three groups:

- **Merged neighbours:** adjacent roofs separated only by a thin gap or shadow come out as one blob.
- **False positives on open ground:** bare sand, shaded tree canopy and partly built sites are labelled as buildings.
- **Large multi-part roofs:** long roofs made of several parallel units are fused in the middle and ragged at the edges.

The baselines err in opposite directions: Decision Tree and Random Forest over-detect (high recall, precision below 0.5), while DeepLabV3 and FPN-ResNet-18 under-detect (precision about 0.75, recall below 0.45). The U-Net keeps precision and recall both near 0.77.

---

## Limitations and Notes

- The test set is small (20 tiles) and every result comes from a single seed.
- Baseline training budgets are much smaller than the U-Net's (see [Results](#results)).
- Random pixel sampling in the Decision Tree and Random Forest training is not seeded, so their scores vary slightly between runs.
- `UNet.use_checkpointing()` in `src/unet_arch.py` is not functional and was not used for any reported result.
- The README of an earlier version listed an Otsu-threshold baseline; it is not part of the final comparison.

---


## Citation

If you use the dataset, please cite:

> S. Ji, S. Wei, M. Lu. *Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set.* IEEE Transactions on Geoscience and Remote Sensing, 2019.