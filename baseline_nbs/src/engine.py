import os
import cv2
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import *
from src.dataset import WHUDataset, extract_features
from src.models.traditional import get_traditional_model
from src.models.deep import get_deep_model

def train_traditional(model_name, split_file):
    print(f"Training Traditional Model: {model_name}...")
    with open(split_file, 'r') as f:
        train_files = [line.strip() for line in f.readlines() if line.strip()]
        
    X_train, y_train = [], []
    for filename in train_files:
        img = cv2.imread(os.path.join(IMAGE_DIR, filename))
        mask = cv2.imread(os.path.join(LABEL_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue
        
        features = extract_features(img)
        labels = (mask.ravel() > 127).astype(np.uint8)
        
        bg_indices = np.where(labels == 0)[0]
        fg_indices = np.where(labels == 1)[0]
        
        if len(bg_indices) > SAMPLES_PER_CLASS: bg_indices = np.random.choice(bg_indices, SAMPLES_PER_CLASS, replace=False)
        if len(fg_indices) > SAMPLES_PER_CLASS: fg_indices = np.random.choice(fg_indices, SAMPLES_PER_CLASS, replace=False)
        
        indices = np.concatenate([bg_indices, fg_indices])
        np.random.shuffle(indices)
        if len(indices) > 0:
            X_train.append(features[indices])
            y_train.append(labels[indices])
            
    X_train, y_train = np.vstack(X_train), np.hstack(y_train)
    model = get_traditional_model(model_name)
    model.fit(X_train, y_train)
    
    save_path = os.path.join(MODEL_OUT_DIR, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def test_traditional(model_name, split_file):
    model = joblib.load(os.path.join(MODEL_OUT_DIR, f"{model_name}.pkl"))
    pred_dir = os.path.join(PRED_OUT_DIR, model_name)
    os.makedirs(pred_dir, exist_ok=True)
    
    with open(split_file, 'r') as f:
        test_files = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Testing {model_name}...")
    for filename in test_files:
        img = cv2.imread(os.path.join(IMAGE_DIR, filename))
        if img is None: continue
        preds = model.predict(extract_features(img))
        pred_mask = (preds.reshape((512, 512)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(pred_dir, filename), pred_mask)
    return pred_dir

def train_deep(model_name, split_file):
    print(f"Training {model_name} on {DEVICE}...")
    dataloader = DataLoader(WHUDataset(split_file), batch_size=BATCH_SIZE, shuffle=True)
    model = get_deep_model(model_name).to(DEVICE)
    criterion, optimizer = nn.BCEWithLogitsLoss(), optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for imgs, masks, _ in dataloader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            if isinstance(outputs, dict): outputs = outputs['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    save_path = os.path.join(MODEL_OUT_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def test_deep(model_name, split_file):
    pred_dir = os.path.join(PRED_OUT_DIR, model_name)
    os.makedirs(pred_dir, exist_ok=True)
    
    model = get_deep_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_OUT_DIR, f"{model_name}.pth"), map_location=DEVICE))
    model.eval()
    
    dataloader = DataLoader(WHUDataset(split_file), batch_size=1, shuffle=False)
    print(f"Testing {model_name}...")
    with torch.no_grad():
        for imgs, _, filenames in dataloader:
            outputs = model(imgs.to(DEVICE))
            if isinstance(outputs, dict): outputs = outputs['out']
            preds = torch.sigmoid(outputs).cpu().numpy()[0, 0]
            pred_mask = (preds > 0.5).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(pred_dir, filenames[0]), pred_mask)
    return pred_dir