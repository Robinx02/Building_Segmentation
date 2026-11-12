import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "satellite_dataset_I")
IMAGE_DIR = os.path.join(DATA_DIR, "image")
LABEL_DIR = os.path.join(DATA_DIR, "label")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_OUT_DIR = os.path.join(OUTPUT_DIR, "models")
PRED_OUT_DIR = os.path.join(OUTPUT_DIR, "predictions")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(PRED_OUT_DIR, exist_ok=True)

# Traditional ML Parameters
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
SAMPLES_PER_CLASS = 2500  # Enforces class balancing

# Deep Learning Parameters
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")