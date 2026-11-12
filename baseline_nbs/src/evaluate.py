import cv2
import os
import csv
import numpy as np
from src.config import LABEL_DIR, METRICS_CSV

def compute_basic_metrics(pred, gt):
    p, t = (pred > 127).astype(np.uint8).ravel(), (gt > 127).astype(np.uint8).ravel()
    tp, fp, fn = np.sum(p * t), np.sum(p * (1 - t)), np.sum((1 - p) * t)
    return (
        tp / (tp + fp + fn + 1e-8),          # IoU
        2 * tp / (2 * tp + fp + fn + 1e-8),  # Dice
        tp / (tp + fp + 1e-8),               # Precision
        tp / (tp + fn + 1e-8)                # Recall
    )

def compute_boundary_f1(pred, gt, tolerance=2):
    gt_edges, pred_edges = cv2.Canny(gt, 50, 150), cv2.Canny(pred, 50, 150)
    kernel = np.ones((tolerance, tolerance), np.uint8)
    gt_dilated, pred_dilated = cv2.dilate(gt_edges, kernel), cv2.dilate(pred_edges, kernel)
    
    tp_p = np.logical_and(pred_edges > 0, gt_dilated > 0).sum()
    fp_p = (pred_edges > 0).sum() - tp_p
    precision = tp_p / (tp_p + fp_p + 1e-8)
    
    tp_r = np.logical_and(gt_edges > 0, pred_dilated > 0).sum()
    fn_r = (gt_edges > 0).sum() - tp_r
    recall = tp_r / (tp_r + fn_r + 1e-8)
    
    return 2 * precision * recall / (precision + recall + 1e-8)

def run_evaluation(pred_dir, split_file, model_name):
    with open(split_file, 'r') as f:
        test_files = [line.strip() for line in f.readlines() if line.strip()]
        
    metrics = {"iou": [], "dice": [], "precision": [], "recall": [], "boundary_f1": []}
    for filename in test_files:
        gt = cv2.imread(os.path.join(LABEL_DIR, filename), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_dir, filename), cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None: continue
            
        iou, dice, prec, rec = compute_basic_metrics(pred, gt)
        metrics["iou"].append(iou)
        metrics["dice"].append(dice)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["boundary_f1"].append(compute_boundary_f1(pred, gt))
        
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    file_exists = os.path.isfile(METRICS_CSV)
    with open(METRICS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Method', 'IoU', 'Dice', 'Precision', 'Recall', 'Boundary F1'])
        writer.writerow([model_name, f"{avg_metrics['iou']:.4f}", f"{avg_metrics['dice']:.4f}", 
                        f"{avg_metrics['precision']:.4f}", f"{avg_metrics['recall']:.4f}", f"{avg_metrics['boundary_f1']:.4f}"])