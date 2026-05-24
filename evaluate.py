import yaml
import torch
import os
import re
import numpy as np
from tqdm import tqdm

from models import DSUnet
from data.dataset import LaneSegmentationDataset
from data.transforms import get_val_transforms
from torch.utils.data import DataLoader
from utils import (get_metrics, load_checkpoint, compute_confusion_matrix, 
                   get_metrics_from_conf_matrix, compute_pr_curve_data,
                   plot_confusion_matrix, plot_precision_recall_curve)

def evaluate(config_path, checkpoint_path, save_dir='results'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    num_classes = config['model']['num_classes']
    
    # Auto-detect latest checkpoint if the specified path doesn't exist
    if not os.path.exists(checkpoint_path):
        base_dir = os.path.dirname(checkpoint_path)
        filename = os.path.basename(checkpoint_path)
        if not base_dir or base_dir == '':
            base_dir = "checkpoints"
            
        max_num = 0
        latest_dir = None
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    match = re.match(r"^checkpoint(\d+)$", item)
                    if match:
                        num = int(match.group(1))
                        # Only select if the target file actually exists in this folder
                        if os.path.exists(os.path.join(base_dir, item, filename)):
                            if num > max_num:
                                max_num = num
                                latest_dir = os.path.join(base_dir, item)
        if latest_dir is not None:
            possible_path = os.path.join(latest_dir, filename)
            print(f"Checkpoint not found at '{checkpoint_path}'. Automatically using the latest checkpoint found at: '{possible_path}'")
            checkpoint_path = possible_path
    
    # Load dataset
    val_transform = get_val_transforms(
        config['dataset']['image_height'], 
        config['dataset']['image_width']
    )
    test_dataset = LaneSegmentationDataset(
        images_dir=config['dataset']['test_images_dir'],
        json_path=config['dataset']['test_json_path'],
        img_height=config['dataset']['image_height'],
        img_width=config['dataset']['image_width'],
        transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Width Scaling / Alpha
    width_multiplier = config['model'].get('width_multiplier', config['model'].get('alpha', 1.0))
    print(f"Model initialization: Width Multiplier (Alpha) = {width_multiplier}")
    
    # Initialize and load model
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=num_classes,
        dropout=config['model'].get('dropout', 0.5),
        width_multiplier=width_multiplier
    ).to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    model.eval()
    model.switch_to_deploy()
    
    conf_matrix = np.zeros((num_classes, num_classes))
    
    # PR Curve accumulation
    thresholds = np.linspace(0.0, 1.0, 21)
    num_thresholds = len(thresholds)
    total_tp = np.zeros((num_classes, num_thresholds))
    total_tp_plus_fp = np.zeros((num_classes, num_thresholds))
    total_gt = np.zeros(num_classes)
    
    print("Evaluating...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            
            # Compute argmax and update confusion matrix incrementally
            preds = torch.argmax(outputs, dim=1)
            conf_matrix += compute_confusion_matrix(preds, masks, num_classes)
            
            # PR curve accumulation
            probs = torch.softmax(outputs, dim=1)
            tp, tp_plus_fp, gt = compute_pr_curve_data(probs, masks, num_classes, thresholds)
            total_tp += tp
            total_tp_plus_fp += tp_plus_fp
            total_gt += gt
            
    metrics = get_metrics_from_conf_matrix(conf_matrix)
    print(f"Evaluation Results - mIoU: {metrics['mIoU']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")
    
    # Compute precision and recall curves
    precision_curve = np.zeros((num_classes, num_thresholds))
    recall_curve = np.zeros((num_classes, num_thresholds))
    for c in range(num_classes):
        precision_curve[c] = total_tp[c] / np.maximum(total_tp_plus_fp[c], 1e-6)
        recall_curve[c] = total_tp[c] / np.maximum(total_gt[c], 1e-6)
        
    os.makedirs(save_dir, exist_ok=True)
    
    if num_classes == 9:
        class_names = ['background', 'continuous white', 'continuous yellow', 'dashed', 'double continuous yellow', 'main-lane', 'other-lane', 'turn-lane', 'vehicle']
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    # Save confusion matrices (both raw and normalized)
    plot_confusion_matrix(conf_matrix, class_names, save_dir, epoch=None)
    
    # Save the Precision-Recall curve
    plot_precision_recall_curve(precision_curve, recall_curve, class_names, save_dir)
    
    print(f"Saved evaluation plots (confusion_matrix.png, confusion_matrix_normalized.png, precision_recall_curve.png) to {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DSUnet Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth', help='Path to checkpoint file')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save evaluation plots')
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, args.save_dir)
