"""
metrics.py - IoU, Dice/F1, precision, recall and Boundary F1 for binary masks (numpy / OpenCV only).

All functions take binary uint8/bool arrays of shape (H, W) (1 = building).
"""
import csv
import time
from pathlib import Path

import cv2
import numpy as np

EPS = 1e-7


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def region_metrics_from_counts(tp, fp, fn):
    """Works for one image or for pooled counts over a whole split."""
    iou = (tp + EPS) / (tp + fp + fn + EPS)            # both empty -> 1.0
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    return dict(iou=iou, dice=dice, precision=precision, recall=recall)


def boundary_map(mask):
    """1-pixel-wide inner boundary of a binary mask (mask minus its erosion)."""
    m = mask.astype(np.uint8)
    # default border handling: the image frame itself is NOT counted as a building edge
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return (m - er).astype(bool)


def boundary_f1(pred, gt, tol=3):

    bp, bg = boundary_map(pred), boundary_map(gt)
    n_p, n_g = int(bp.sum()), int(bg.sum())
    if n_p == 0 and n_g == 0:
        return 1.0, 1.0, 1.0           # nothing to find and nothing predicted
    if n_p == 0 or n_g == 0:
        return 0.0, 0.0, 0.0
    # distance of every pixel to the nearest boundary pixel of the other mask
    dist_to_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pr = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((dist_to_gt[bp] <= tol).sum()) / n_p
    rec = float((dist_to_pr[bg] <= tol).sum()) / n_g
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


def evaluate_pair(pred, gt, tol=3):
    """All metrics for one image + the raw counts (so you can also pool across images)."""
    tp, fp, fn, tn = confusion(pred, gt)
    out = region_metrics_from_counts(tp, fp, fn)
    bf1, bp_, br_ = boundary_f1(pred, gt, tol)
    out.update(boundary_f1=bf1, boundary_precision=bp_, boundary_recall=br_, tp=tp, fp=fp, fn=fn, tn=tn, gt_fraction=(tp + fn) / (tp + fp + fn + tn))
    return out


# split-level helpers
_MEAN_KEYS = ["iou", "dice", "precision", "recall", "boundary_f1", "boundary_precision", "boundary_recall"]


def evaluate_masks(names, preds, gts, tol=3):
    """names/preds/gts: parallel lists. Returns one metric dict per image (with an 'image' key)."""
    rows = []
    for n, p, g in zip(names, preds, gts):
        r = evaluate_pair(p, g, tol)
        r["image"] = n
        rows.append(r)
    return rows


def summarize(rows):
    """Mean of per-image metrics (+ 'pooled_*' = metrics from summed pixel counts)."""
    out = {k: float(np.mean([r[k] for r in rows])) for k in _MEAN_KEYS}
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    out.update({f"pooled_{k}": v for k, v in region_metrics_from_counts(tp, fp, fn).items()})
    out["n_images"] = len(rows)
    return out


def write_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def append_csv(path, row):
    """Append one dict as a row; adds new columns automatically if the header has to grow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
    keys = list(old[0].keys()) if old else []
    for k in row:
        if k not in keys:
            keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in old + [row]:
            w.writerow({k: r.get(k, "") for k in keys})


def result_row(method, split, summary, **extra):
    """One line of the final results table (use this from baselines AND from train.py)."""
    row = dict(method=method, split=split, **summary, **extra)
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return row