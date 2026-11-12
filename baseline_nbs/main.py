import argparse
import os
from src.config import SPLITS_DIR
from src.engine import train_traditional, test_traditional, train_deep, test_deep
from src.evaluate import run_evaluation

TRADITIONAL_MODELS = ['random_forest', 'decision_tree']
DEEP_MODELS = ['deeplabv3', 'resnet18']

def main():
    parser = argparse.ArgumentParser(description="Building Segmentation Pipeline")
    parser.add_argument('--model', type=str, required=True, choices=TRADITIONAL_MODELS + DEEP_MODELS)
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--split', type=str, default='train.txt')
    args = parser.parse_args()
    
    split_path = os.path.join(SPLITS_DIR, args.split)
    if not os.path.exists(split_path):
        print(f"Error: Split file {split_path} not found.")
        return

    is_traditional = args.model in TRADITIONAL_MODELS

    if args.mode == 'train':
        train_traditional(args.model, split_path) if is_traditional else train_deep(args.model, split_path)
            
    elif args.mode == 'test':
        pred_dir = test_traditional(args.model, split_path) if is_traditional else test_deep(args.model, split_path)
        # Automatically trigger evaluation logging
        run_evaluation(pred_dir, split_path, args.model)
        print(f"Evaluation complete. Metrics added to outputs/metrics.csv")

if __name__ == "__main__":
    main()