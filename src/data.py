'''
data.py WHU building dataset loading + light augmentation

Leakage rules enforced here:
  * augmentation + random cropping ONLY for the train split (crops never leave their parent image,
    and the split is done per image/group BEFORE cropping)
  * normalisation statistics are computed from the train split only

'''

import random

from pathlib import Path

import cv2 
import numpy as np
import torch

from torch.utils.data import Dataset

IMG_EXT = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_manifest(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def _find_label(label_dir: Path, stem: str):
    for ext in IMG_EXT + [e.upper() for e in IMG_EXT]:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no label for {stem} in {label_dir}")


def load_pair(root, name):
    root = Path(root)
    img = cv2.imread(str(root / "image" / name), cv2.IMREAD_COLOR)          # 8-bit BGR
    if img is None:
        raise IOError(f"cannot read image {name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lbl = cv2.imread(str(_find_label(root / "label", Path(name).stem)), cv2.IMREAD_GRAYSCALE)
    if lbl is None:
        raise IOError(f"cannot read label for {name}")
    mask = (lbl > 0).astype(np.uint8)                                        # works for 0/1 and 0/255
    assert img.shape[:2] == mask.shape, f"size mismatch image/label for {name}"
    return img, mask


def compute_mean_std(root, names):
    """Per-channel mean/std over the given (TRAIN) images, in [0,1] units."""
    s, s2, n = np.zeros(3), np.zeros(3), 0
    for name in names:
        img, _ = load_pair(root, name)
        x = img.reshape(-1, 3).astype(np.float64) / 255.0
        s += x.sum(0)
        s2 += (x ** 2).sum(0)
        n += x.shape[0]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return mean.tolist(), std.tolist()


class WHUDataset(Dataset):
    def __init__(self, root, names, train=False, crop=0, mean=None, std=None,rot_prob=0.3, max_angle=15, repeat=1):
        self.root, self.names, self.train, self.crop = Path(root), list(names), train, crop
        self.rot_prob, self.max_angle = rot_prob, max_angle
        self.repeat = repeat          # >1: see every image several times per epoch (different random crops)
        self.mean = np.array(mean if mean is not None else IMAGENET_MEAN, np.float32)
        self.std = np.array(std if std is not None else IMAGENET_STD, np.float32)
        self.data = [load_pair(self.root, n) for n in self.names]

    def __len__(self):
        return len(self.names) * self.repeat


### Now the augmentation part here

    def _augment(self, img, mask):
        H, W = mask.shape
        if self.crop and self.crop < min(H, W):
            y = random.randint(0, H - self.crop)
            x = random.randint(0, W - self.crop)
            img, mask = img[y:y + self.crop, x:x + self.crop], mask[y:y + self.crop, x:x + self.crop]
        if random.random() < 0.5:                                  # horizontal flip
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:                                  # vertical flip
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)                                   # 90-degree rotations (lossless)
        if k:
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        if random.random() < self.rot_prob:                        # small free rotation
            h, w = mask.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-self.max_angle, self.max_angle), 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        alpha, beta = random.uniform(0.8, 1.2), random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, i):
        i = i % len(self.names)
        img, mask = self.data[i]
        if self.train:
            img, mask = self._augment(img, mask)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return x, y, self.names[i]

