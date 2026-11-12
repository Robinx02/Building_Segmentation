# rescore_baselines.py  (run from the U-Net project root; adjust the 3 paths)
import cv2, numpy as np
from metrics import evaluate_masks, summarize

LABELS = "baseline_nbs/data/raw/satellite_dataset_I/label"
PREDS  = "baseline_nbs/outputs/predictions"

names  = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
for m in ["decision_tree", "random_forest", "resnet18", "deeplabv3"]:
    P = [(cv2.imread(f"{PREDS}/{m}/{n}", 0) > 127).astype(np.uint8) for n in names]
    G = [(cv2.imread(f"{LABELS}/{n}", 0) > 127).astype(np.uint8) for n in names]
    s = summarize(evaluate_masks(names, P, G, tol=3))
    print(m, *[f"{s[k]:.4f}" for k in ("iou", "dice", "precision", "recall", "boundary_f1")])