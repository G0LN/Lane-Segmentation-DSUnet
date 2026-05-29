import os
import re
import yaml
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models import DSUnet
from data import get_dataloaders
from utils import (get_criterion, get_metrics, compute_confusion_matrix, 
                    get_metrics_from_conf_matrix, Logger, set_seed, save_checkpoint,
                    compute_pr_curve_data, plot_precision_recall_curve)
from utils.plotters import plot_training_curves, plot_confusion_matrix

class Trainer:
    """
    Object-Oriented Trainer class for B-DSUnet.
    Encapsulates all training state, dataloaders, model components, logging,
    checkpoints, and evaluation pipelines to ensure modular, readable code.
    """
    def __init__(self, config_path="configs/default.yaml", resume=False):
        self.config_path = config_path
        self.resume = resume
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        set_seed(self.config['training']['seed'])
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else "cpu")
        self.num_classes = self.config['model']['num_classes']
        self.num_epochs = self.config['training']['epochs']
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', None)
        
        # Core training state
        self.start_epoch = 0
        self.best_miou = 0.0
        self.early_stopping_counter = 0
        
        # History metrics for plotting
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_miou': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': []
        }
        
        # Setup directories and loader
        self._setup_directories()
        self._setup_data()
        self._setup_model()
        self._setup_logger()
        
        if self.resume:
            self._load_checkpoint_if_exists()

    def _get_latest_checkpoint_dir(self, base_dir):
        max_num = 0
        latest_dir = None
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    match = re.match(r"^checkpoint(\d+)$", item)
                    if match:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
                            latest_dir = os.path.join(base_dir, item)
        return latest_dir, max_num

    def _setup_directories(self):
        base_save_dir = self.config['training']['save_dir']
        base_log_dir = self.config['training']['log_dir']
        
        checkpoint_path = None
        if self.resume:
            latest_dir, max_num = self._get_latest_checkpoint_dir(base_save_dir)
            if latest_dir is not None and os.path.exists(os.path.join(latest_dir, "checkpoint.pth")):
                self.save_dir = latest_dir
                self.log_dir = os.path.join(base_log_dir, f"log{max_num}")
                self.checkpoint_path = os.path.join(latest_dir, "checkpoint.pth")
                print(f"Resuming training in existing directories:")
                print(f" -> Save directory: {self.save_dir}")
                print(f" -> Log directory: {self.log_dir}")
                return
            else:
                print(f"No valid checkpoint found to resume under '{base_save_dir}'. Starting fresh run...")
                
        # Start a new training run directory
        _, max_num = self._get_latest_checkpoint_dir(base_save_dir)
        next_num = max_num + 1
        self.save_dir = os.path.join(base_save_dir, f"checkpoint{next_num}")
        self.log_dir = os.path.join(base_log_dir, f"log{next_num}")
        self.checkpoint_path = None
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Created new training directories:")
        print(f" -> Save directory: {self.save_dir}")
        print(f" -> Log directory: {self.log_dir}")

    def _setup_data(self):
        self.train_loader, self.val_loader = get_dataloaders(self.config)

    def _setup_model(self):
        # Scale channels according to width_multiplier (alpha)
        width_multiplier = self.config['model'].get('width_multiplier', self.config['model'].get('alpha', 1.0))
        print(f"Model initialization: Width Multiplier (Alpha) = {width_multiplier}")
        
        self.model = DSUnet(
            in_channels=self.config['model']['in_channels'], 
            num_classes=self.num_classes,
            dropout=self.config['model'].get('dropout', 0.5),
            width_multiplier=width_multiplier
        ).to(self.device)
        
        # Loss, Optimizer & Scheduler
        self.criterion = get_criterion(self.config['loss']['type'], num_classes=self.num_classes, device=self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'], 
            weight_decay=self.config['training']['weight_decay']
        )

    def _setup_logger(self):
        self.logger = Logger(self.log_dir)

    def _load_checkpoint_if_exists(self):
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"\nFound existing checkpoint at {self.checkpoint_path}. Resuming...")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']
                self.best_miou = checkpoint['best_miou']
                self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
                if 'history' in checkpoint:
                    self.history = checkpoint['history']
                patience_str = f"/{self.early_stopping_patience}" if self.early_stopping_patience is not None else ""
                print(f"Successfully resumed from epoch {self.start_epoch} | Best mIoU: {self.best_miou:.4f} (Counter: {self.early_stopping_counter}{patience_str})")
            except Exception as e:
                print(f"Could not load checkpoint: {e}. Starting fresh instead.")

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        valid_batches = 0
        for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Warning] NaN/Inf Loss detected at Epoch {epoch+1}! Skipping this batch.")
                self.optimizer.zero_grad()
                continue
                
            loss.backward()
            
            # Verify gradients are finite before step
            is_finite = True
            for p in self.model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        is_finite = False
                        break
            
            if is_finite:
                # Prevent Gradient Explosion (NaN loss) using Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item()
                valid_batches += 1
            else:
                print(f"\n[Warning] NaN/Inf Gradients detected at Epoch {epoch+1}! Skipping optimizer step to prevent weight corruption.")
                self.optimizer.zero_grad()
                
        if valid_batches == 0:
            return 0.0
        return running_loss / valid_batches

    def _validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        conf_matrix = np.zeros((self.num_classes, self.num_classes))
        
        # PR Curve threshold initialization
        thresholds = np.linspace(0.0, 1.0, 21)
        num_thresholds = len(thresholds)
        total_tp = np.zeros((self.num_classes, num_thresholds))
        total_tp_plus_fp = np.zeros((self.num_classes, num_thresholds))
        total_gt = np.zeros(self.num_classes)
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                
                # Get prediction labels
                preds = torch.argmax(outputs, dim=1)
                conf_matrix += compute_confusion_matrix(preds, masks, self.num_classes)
                
                # Accumulate for precision-recall curves
                probs = torch.softmax(outputs, dim=1)
                tp, tp_plus_fp, gt = compute_pr_curve_data(probs, masks, self.num_classes, thresholds)
                total_tp += tp
                total_tp_plus_fp += tp_plus_fp
                total_gt += gt
                
        metrics = get_metrics_from_conf_matrix(conf_matrix)
        
        # Calculate precision & recall curve arrays
        precision_curve = np.zeros((self.num_classes, num_thresholds))
        recall_curve = np.zeros((self.num_classes, num_thresholds))
        for c in range(self.num_classes):
            precision_curve[c] = total_tp[c] / np.maximum(total_tp_plus_fp[c], 1e-6)
            recall_curve[c] = total_tp[c] / np.maximum(total_gt[c], 1e-6)
            
        metrics['precision_curve'] = precision_curve
        metrics['recall_curve'] = recall_curve
        
        return running_loss / len(self.val_loader), metrics

    def fit(self):
        print(f"Starting training on {self.device} for {self.num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")
            
            # 1. Run Train and Val loops
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_metrics['mIoU']:.4f}")
            print(f"Precision: {val_metrics['Precision']:.4f} | Recall: {val_metrics['Recall']:.4f} | F1: {val_metrics['F1']:.4f}")
            
            # 2. Log scalars
            self.logger.log_scalar("Loss/train", train_loss, epoch)
            self.logger.log_scalar("Loss/val", val_loss, epoch)
            self.logger.log_scalar("Metrics/mIoU", val_metrics['mIoU'], epoch)
            
            # 3. Save to history dictionary
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_miou'].append(val_metrics['mIoU'])
            self.history['val_precision'].append(val_metrics['Precision'])
            self.history['val_recall'].append(val_metrics['Recall'])
            self.history['val_f1'].append(val_metrics['F1'])
            
            # 4. Generate dynamic curves
            plot_training_curves(self.history, self.log_dir)
            
            # 5. Check validation improvement
            is_best = val_metrics['mIoU'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['mIoU']
                self.early_stopping_counter = 0
                print(f"[Best] New best validation mIoU: {self.best_miou:.4f}! Saving best model...")
                
                # Class names setup
                if self.num_classes == 9:
                    class_names = ['background', 'continuous white', 'continuous yellow', 'dashed', 'double continuous yellow', 'main-lane', 'other-lane', 'turn-lane', 'vehicle']
                else:
                    class_names = [f"Class_{i}" for i in range(self.num_classes)]
                    
                # Save confusion matrix and PR curves ONLY for best result to save disk space
                plot_confusion_matrix(val_metrics['ConfusionMatrix'], class_names, self.log_dir, epoch=None)
                plot_precision_recall_curve(val_metrics['precision_curve'], val_metrics['recall_curve'], class_names, self.log_dir)
            else:
                self.early_stopping_counter += 1
                patience_msg = f"/{self.early_stopping_patience}" if self.early_stopping_patience is not None else ""
                print(f"No improvement in validation mIoU for {self.early_stopping_counter}{patience_msg} consecutive epochs.")
                
            # 6. Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_miou': self.best_miou,
                'optimizer': self.optimizer.state_dict(),
                'history': self.history,
                'early_stopping_counter': self.early_stopping_counter
            }, is_best, self.save_dir)
            
            # 7. Early stopping validation
            if self.early_stopping_patience is not None and self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement in validation mIoU for {self.early_stopping_patience} consecutive epochs. Stopping training early!")
                break
                
        print(f"\nTraining completed! Best Validation mIoU reached: {self.best_miou:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DSUnet Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint directory')
    args = parser.parse_args()
    
    trainer = Trainer(config_path=args.config, resume=args.resume)
    trainer.fit()
