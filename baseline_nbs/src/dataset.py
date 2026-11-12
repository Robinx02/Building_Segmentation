import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern
from src.config import IMAGE_DIR, LABEL_DIR, LBP_RADIUS, LBP_N_POINTS

class WHUDataset(Dataset):
    def __init__(self, split_file):
        with open(split_file, 'r') as f:
            self.files = [line.strip() for line in f.readlines() if line.strip()]
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        img = cv2.imread(os.path.join(IMAGE_DIR, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        mask = cv2.imread(os.path.join(LABEL_DIR, filename), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, filename

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    features = np.vstack([
        image[:, :, 0].ravel(),
        image[:, :, 1].ravel(),
        image[:, :, 2].ravel(),
        lbp.ravel()
    ]).T
    return features