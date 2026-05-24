import os
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    conf_matrix = np.zeros((num_classes, num_classes))
    
    # PR Curve accumulation
    thresholds = np.linspace(0.0, 1.0, 21)
    num_thresholds = len(thresholds)
    total_tp = np.zeros((num_classes, num_thresholds))
    total_tp_plus_fp = np.zeros((num_classes, num_thresholds))
    total_gt = np.zeros(num_classes)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
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
    
    # Compute precision and recall curves
    precision_curve = np.zeros((num_classes, num_thresholds))
    recall_curve = np.zeros((num_classes, num_thresholds))
    for c in range(num_classes):
        precision_curve[c] = total_tp[c] / np.maximum(total_tp_plus_fp[c], 1e-6)
        recall_curve[c] = total_tp[c] / np.maximum(total_gt[c], 1e-6)
        
    metrics['precision_curve'] = precision_curve
    metrics['recall_curve'] = recall_curve
    
    return running_loss / len(dataloader), metrics

def main(config_path="configs/default.yaml", resume=False):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['training']['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    num_classes = config['model']['num_classes']
    
    # Dataloaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Width Scaling / Alpha
    width_multiplier = config['model'].get('width_multiplier', config['model'].get('alpha', 1.0))
    print(f"Model initialization: Width Multiplier (Alpha) = {width_multiplier}")
    
    # Determine save and log directories dynamically
    base_save_dir = config['training']['save_dir']
    base_log_dir = config['training']['log_dir']
    
    import re
    def get_latest_checkpoint_dir(base_dir):
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

    checkpoint_path = None
    if resume:
        latest_dir, max_num = get_latest_checkpoint_dir(base_save_dir)
        if latest_dir is not None and os.path.exists(os.path.join(latest_dir, "checkpoint.pth")):
            save_dir = latest_dir
            log_dir = os.path.join(base_log_dir, f"log{max_num}")
            checkpoint_path = os.path.join(latest_dir, "checkpoint.pth")
            print(f"Resuming training in existing directories:")
            print(f" -> Save directory: {save_dir}")
            print(f" -> Log directory: {log_dir}")
        else:
            # Fallback to new training
            print(f"No valid checkpoint found to resume under '{base_save_dir}'. Starting a new training run...")
            latest_dir, max_num = get_latest_checkpoint_dir(base_save_dir)
            next_num = max_num + 1
            save_dir = os.path.join(base_save_dir, f"checkpoint{next_num}")
            log_dir = os.path.join(base_log_dir, f"log{next_num}")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created new directories:")
            print(f" -> Save directory: {save_dir}")
            print(f" -> Log directory: {log_dir}")
    else:
        # Start a new training run
        latest_dir, max_num = get_latest_checkpoint_dir(base_save_dir)
        next_num = max_num + 1
        save_dir = os.path.join(base_save_dir, f"checkpoint{next_num}")
        log_dir = os.path.join(base_log_dir, f"log{next_num}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created new directories for a fresh run:")
        print(f" -> Save directory: {save_dir}")
        print(f" -> Log directory: {log_dir}")

    # Model
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=num_classes,
        dropout=config['model'].get('dropout', 0.5),
        width_multiplier=width_multiplier
    ).to(device)
    
    # Loss and Optimizer
    criterion = get_criterion(config['loss']['type'], num_classes=num_classes, device=device)
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'])
    
    # Logger
    logger = Logger(log_dir)
    
    # History for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'val_miou': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': []
    }
    
    start_epoch = 0
    best_miou = 0.0
    num_epochs = config['training']['epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', None)
    early_stopping_counter = 0
    
    # Auto-resume logic
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_miou = checkpoint['best_miou']
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            if 'history' in checkpoint:
                history = checkpoint['history']
            patience_str = f"/{early_stopping_patience}" if early_stopping_patience is not None else ""
            print(f"Successfully resumed from epoch {start_epoch} with best mIoU {best_miou:.4f} (Early Stopping: {early_stopping_counter}{patience_str})")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            
    print(f"Starting training for {num_epochs} epochs on {device}...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, num_classes)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_metrics['mIoU']:.4f}")
        print(f"Precision: {val_metrics['Precision']:.4f} | Recall: {val_metrics['Recall']:.4f} | F1: {val_metrics['F1']:.4f}")
        
        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)
        logger.log_scalar("Metrics/mIoU", val_metrics['mIoU'], epoch)
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_metrics['mIoU'])
        history['val_precision'].append(val_metrics['Precision'])
        history['val_recall'].append(val_metrics['Recall'])
        history['val_f1'].append(val_metrics['F1'])
        
        # Plot curves dynamically
        plot_training_curves(history, log_dir)
        
        # Plot confusion matrix only periodically or at the end to save time
        # (Removed periodic plotting as per user request to only save for best result)
        
        # Save checkpoint and early stopping check
        is_best = val_metrics['mIoU'] > best_miou
        if is_best:
            best_miou = val_metrics['mIoU']
            early_stopping_counter = 0
            print(f"✨ New best validation mIoU: {best_miou:.4f}! Saving best model...")
            
            # Save confusion matrix and precision-recall curve ONLY for the best epoch
            if num_classes == 9:
                class_names = ['background', 'continuous white', 'continuous yellow', 'dashed', 'double continuous yellow', 'main-lane', 'other-lane', 'turn-lane', 'vehicle']
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]
                
            # Plot and save both raw and normalized confusion matrices
            plot_confusion_matrix(val_metrics['ConfusionMatrix'], class_names, log_dir, epoch=None)
            
            # Plot and save the Precision-Recall curve
            plot_precision_recall_curve(val_metrics['precision_curve'], val_metrics['recall_curve'], class_names, log_dir)
        else:
            early_stopping_counter += 1
            patience_msg = f"/{early_stopping_patience}" if early_stopping_patience is not None else ""
            print(f"No improvement in validation mIoU for {early_stopping_counter}{patience_msg} consecutive epochs.")
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_miou': best_miou,
            'optimizer': optimizer.state_dict(),
            'history': history,
            'early_stopping_counter': early_stopping_counter
        }, is_best, save_dir=save_dir)
        
        # Trigger early stopping if patience reached
        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered! Training stopped because validation mIoU did not improve for {early_stopping_patience} consecutive epochs.")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DSUnet")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()
    
    main(config_path=args.config, resume=args.resume)
