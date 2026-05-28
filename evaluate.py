import os
import re
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import DSUnet
from data.dataset import LaneSegmentationDataset
from data.transforms import get_val_transforms
from utils import (get_metrics, load_checkpoint, compute_confusion_matrix, 
                    get_metrics_from_conf_matrix, compute_pr_curve_data,
                    plot_confusion_matrix, plot_precision_recall_curve)

class Evaluator:
    """
    Object-Oriented Evaluator class for B-DSUnet.
    Encapsulates all evaluation pipelines, dataset loaders, models,
    confusion matrix accumulations, and Precision-Recall metrics calculations.
    """
    def __init__(self, config_path="configs/default.yaml", checkpoint_path="checkpoints/model_best.pth", save_dir="results"):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.save_dir = save_dir
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else "cpu")
        self.num_classes = self.config['model']['num_classes']
        
        self._auto_detect_latest_checkpoint()
        self._setup_dataloader()
        self._setup_model()

    def _auto_detect_latest_checkpoint(self):
        # Auto-detect latest checkpoint if the specified path doesn't exist
        if not os.path.exists(self.checkpoint_path):
            base_dir = os.path.dirname(self.checkpoint_path)
            filename = os.path.basename(self.checkpoint_path)
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
                print(f"Checkpoint not found at '{self.checkpoint_path}'. Automatically using the latest found at: '{possible_path}'")
                self.checkpoint_path = possible_path

    def _setup_dataloader(self):
        val_transform = get_val_transforms(
            self.config['dataset']['image_height'], 
            self.config['dataset']['image_width']
        )
        self.test_dataset = LaneSegmentationDataset(
            images_dir=self.config['dataset']['test_images_dir'],
            json_path=self.config['dataset']['test_json_path'],
            img_height=self.config['dataset']['image_height'],
            img_width=self.config['dataset']['image_width'],
            transform=val_transform
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )

    def _setup_model(self):
        width_multiplier = self.config['model'].get('width_multiplier', self.config['model'].get('alpha', 1.0))
        print(f"Model initialization: Width Multiplier (Alpha) = {width_multiplier}")
        
        self.model = DSUnet(
            in_channels=self.config['model']['in_channels'], 
            num_classes=self.num_classes,
            dropout=self.config['model'].get('dropout', 0.5),
            width_multiplier=width_multiplier
        ).to(self.device)
        
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        load_checkpoint(self.checkpoint_path, self.model)
        self.model.eval()
        self.model.switch_to_deploy() # Benchmark actual Deploy mode

    def evaluate(self):
        conf_matrix = np.zeros((self.num_classes, self.num_classes))
        
        # PR Curve threshold initialization
        thresholds = np.linspace(0.0, 1.0, 21)
        num_thresholds = len(thresholds)
        total_tp = np.zeros((self.num_classes, num_thresholds))
        total_tp_plus_fp = np.zeros((self.num_classes, num_thresholds))
        total_gt = np.zeros(self.num_classes)
        
        print("Evaluating...")
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                
                # Update confusion matrix incrementally
                preds = torch.argmax(outputs, dim=1)
                conf_matrix += compute_confusion_matrix(preds, masks, self.num_classes)
                
                # PR curve threshold accumulation
                probs = torch.softmax(outputs, dim=1)
                tp, tp_plus_fp, gt = compute_pr_curve_data(probs, masks, self.num_classes, thresholds)
                total_tp += tp
                total_tp_plus_fp += tp_plus_fp
                total_gt += gt
                
        metrics = get_metrics_from_conf_matrix(conf_matrix)
        print("-" * 50)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("-" * 50)
        print(f"  - Validation mIoU: {metrics['mIoU']:.4f}")
        print(f"  - Accuracy       : {metrics['Accuracy']:.4f}")
        print(f"  - Precision      : {metrics['Precision']:.4f}")
        print(f"  - Recall         : {metrics['Recall']:.4f}")
        print(f"  - F1-Score       : {metrics['F1']:.4f}")
        print("-" * 50)
        
        # Compute precision and recall curve arrays
        precision_curve = np.zeros((self.num_classes, num_thresholds))
        recall_curve = np.zeros((self.num_classes, num_thresholds))
        for c in range(self.num_classes):
            precision_curve[c] = total_tp[c] / np.maximum(total_tp_plus_fp[c], 1e-6)
            recall_curve[c] = total_tp[c] / np.maximum(total_gt[c], 1e-6)
            
        os.makedirs(self.save_dir, exist_ok=True)
        
        if self.num_classes == 9:
            class_names = ['background', 'continuous white', 'continuous yellow', 'dashed', 'double continuous yellow', 'main-lane', 'other-lane', 'turn-lane', 'vehicle']
        else:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]
            
        # Save confusion matrices (both raw and normalized)
        plot_confusion_matrix(conf_matrix, class_names, self.save_dir, epoch=None)
        
        # Save the Precision-Recall curve
        plot_precision_recall_curve(precision_curve, recall_curve, class_names, self.save_dir)
        
        print(f"[Success] Saved evaluation plots (confusion_matrix.png, confusion_matrix_normalized.png, precision_recall_curve.png) under '{self.save_dir}'")
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DSUnet Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth', help='Path to checkpoint file')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save evaluation plots')
    args = parser.parse_args()
    
    evaluator = Evaluator(config_path=args.config, checkpoint_path=args.checkpoint, save_dir=args.save_dir)
    evaluator.evaluate()
