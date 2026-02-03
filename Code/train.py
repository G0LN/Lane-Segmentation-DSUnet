''''
test git
'''
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from tqdm import tqdm
import random
import gc

from DSUnet_lite_model import DSUnet, DeployableDSUnet

# --- STABILITY SETTINGS ---
# Disable CuDNN benchmark to prevent driver crashes on some laptops
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# --- 0. SYSTEM CONFIGURATION ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Enable CUDA launch blocking for better error reporting if needed
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- 1. CONFIGURATION & CONSTANTS ---
NUM_CLASSES = 4
IMG_HEIGHT = 288
IMG_WIDTH = 512

BATCH_SIZE = 8 

EPOCHS = 200
LEARNING_RATE = 1e-5
PATIENCE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. DATASET CLASS ---
class MultiClassLaneDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=(288, 512), augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            mask_path = self.mask_paths[idx]

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0) # Read as grayscale

            if img is None or mask is None:
                raise ValueError(f"Failed to read image: {img_path}")

            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

            if self.augment:
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)

            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1) # [C, H, W]
            
            mask = torch.from_numpy(mask).long()
            img = torch.from_numpy(img)

            return img, mask, img_path
        except Exception as e:
            # Return dummy data to prevent crash
            print(f"Warning: Error loading file {idx}: {e}")
            return torch.zeros((3, self.img_size[0], self.img_size[1])), torch.zeros((self.img_size[0], self.img_size[1])).long(), ""

# --- 3. UTILITIES FOR PATHS ---
def get_mixed_dataset_paths(dir_path):
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        return [], []
    
    all_files = os.listdir(dir_path)
    img_files = [f for f in all_files if f.lower().endswith('.jpg')]
    
    img_paths = []
    mask_paths = []
    
    print(f"🔍 Scanning {dir_path}...")
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        expected_mask_name = f"{base_name}_mask.png"
        
        img_full_path = os.path.join(dir_path, img_file)
        mask_full_path = os.path.join(dir_path, expected_mask_name)
        
        if os.path.exists(mask_full_path):
            img_paths.append(img_full_path)
            mask_paths.append(mask_full_path)
        
    print(f"✅ Found {len(img_paths)} pairs in {os.path.basename(dir_path)}")
    return img_paths, mask_paths

# --- 4. METRICS & ANALYSIS ---
def compute_confusion_matrix_batch(preds, targets, num_classes):
    """Calculates confusion matrix for a single batch."""
    pred_labels = torch.argmax(preds, dim=1).view(-1)
    targets = targets.view(-1)
    
    mask = (targets >= 0) & (targets < num_classes)
    pred_labels = pred_labels[mask]
    targets = targets[mask]
    
    indices = targets * num_classes + pred_labels
    count = torch.bincount(indices, minlength=num_classes**2)
    cm = count.reshape(num_classes, num_classes)
    return cm

def calculate_epoch_metrics(confusion_matrix):
    """
    Calculates detailed metrics from the accumulated Confusion Matrix.
    Returns: mIoU, Accuracy, Mean Precision, Mean Recall, Mean F1
    """
    cm = confusion_matrix.float()
    
    # Pixel Accuracy
    total_pixels = cm.sum()
    correct_pixels = torch.diag(cm).sum()
    accuracy = correct_pixels / (total_pixels + 1e-6)
    
    # Class-wise metrics
    tp = torch.diag(cm)
    row_sum = cm.sum(dim=1) # Ground Truth counts
    col_sum = cm.sum(dim=0) # Predicted counts
    
    # IoU
    union = row_sum + col_sum - tp
    iou = tp / (union + 1e-6)
    miou = iou.mean()
    
    # Precision (TP / TP + FP) -> TP / col_sum
    precision = tp / (col_sum + 1e-6)
    mean_precision = precision.mean()
    
    # Recall (TP / TP + FN) -> TP / row_sum
    recall = tp / (row_sum + 1e-6)
    mean_recall = recall.mean()
    
    # F1 Score per class
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    mean_f1 = f1.mean()
    
    return miou.item(), accuracy.item(), mean_precision.item(), mean_recall.item(), mean_f1.item()

# --- 5. VISUALIZATION & SUMMARY ---
def save_model_summary(model, save_dir):
    """Saves the model architecture summary to a text file."""
    summary_path = os.path.join(save_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(str(model))
        f.write("\n\n--- Model Parameters ---\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
    print(f"✅ Model summary saved to: {summary_path}")

def plot_comprehensive_metrics(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', linestyle='-')
    plt.plot(epochs, history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'convergence_loss.png'))
    plt.close()

    # Plot 2: Detailed Metrics (Acc, Prec, Recall, F1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['val_acc'], label='Accuracy', marker='.')
    plt.plot(epochs, history['val_prec'], label='Precision', marker='.')
    plt.plot(epochs, history['val_recall'], label='Recall', marker='.')
    plt.plot(epochs, history['val_f1'], label='F1-Score', marker='o', linewidth=2)
    plt.title('Validation Performance Metrics over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0-1)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, 'detailed_metrics.png'))
    plt.close()

def plot_confusion_matrices(cm_tensor, classes, save_dir):
    cm = cm_tensor.cpu().numpy()
    
    # 1. Normal Counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Pixel Counts)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_counts.png'))
    plt.close()
    
    # 2. Percentage (Row-normalized)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized %)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_percent.png'))
    plt.close()

# --- 6. MAIN EXECUTION ---
def main():
    # --- DIRECTORY SETUP ---
    # Edit these paths according to your environment
    BASE_OUTPUT_DIR = r'E:\DSUnet-drivable_area_segmentation\Experiments'
    DATASET_ROOT = r'E:\DSUnet-drivable_area_segmentation\Data\Lane-Segmentation-Auto.v2i.png-mask-semantic'
    
    EXPERIMENT_NAME = f"DSUnet_SafetyFP32_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    EXP_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME)
    MODELS_DIR = os.path.join(EXP_DIR, 'models')
    LOGS_DIR = os.path.join(EXP_DIR, 'logs')
    PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
    
    # Create directories only once
    for d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
    VALID_DIR = os.path.join(DATASET_ROOT, 'valid')

    print(f"🚀 Experiment Started: {EXPERIMENT_NAME}")
    print(f"✅ Device: {DEVICE} | Classes: {NUM_CLASSES}")
    print(f"⚙️ Config: FP32 Mode (No AMP) | Batch Size: {BATCH_SIZE} | Workers: 0")

    # --- DATA LOADING ---
    train_imgs, train_masks = get_mixed_dataset_paths(TRAIN_DIR)
    val_imgs, val_masks = get_mixed_dataset_paths(VALID_DIR)
    
    if not train_imgs or not val_imgs:
        print("❌ Error: Images not found. Check paths.")
        return

    train_ds = MultiClassLaneDataset(train_imgs, train_masks, (IMG_HEIGHT, IMG_WIDTH), augment=True)
    val_ds = MultiClassLaneDataset(val_imgs, val_masks, (IMG_HEIGHT, IMG_WIDTH), augment=False)
    
    # num_workers=0 to prevent Windows multiprocessing crash
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # --- MODEL SETUP ---
    model = DSUnet(n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # scaler = GradScaler() # REMOVED FOR STABILITY
    
    # Save Model Summary
    save_model_summary(model, LOGS_DIR)

    # Metrics Storage
    history = {
        'train_loss': [], 'val_loss': [],
        'val_miou': [], 'val_acc': [], 
        'val_prec': [], 'val_recall': [], 'val_f1': []
    }
    
    best_metric = 0.0
    patience_counter = 0
    best_confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long).to(DEVICE)
    
    print("\n🔥 STARTING TRAINING (SAFE MODE)...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN LOOP ---
        model.train()
        train_loss_acc = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            # --- STANDARD FP32 TRAINING (No Autocast) ---
            logits = model(imgs)
            loss = criterion(logits, masks)
            
            # Check for NaN to prevent crash
            if torch.isnan(loss):
                print("⚠️ Warning: Loss is NaN. Skipping batch.")
                del imgs, masks, logits, loss
                continue

            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # --------------------------------------------
            
            train_loss_acc += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # Manually delete variables to free VRAM immediately
            del imgs, masks, logits, loss
            
        avg_train_loss = train_loss_acc / len(train_loader)
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss_acc = 0
        epoch_cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                logits = model(imgs)
                loss = criterion(logits, masks)
                
                val_loss_acc += loss.item()
                
                # Accumulate Confusion Matrix
                batch_cm = compute_confusion_matrix_batch(logits, masks, NUM_CLASSES)
                epoch_cm += batch_cm
                
                del imgs, masks, logits, loss

        avg_val_loss = val_loss_acc / len(val_loader)
        
        # Calculate Epoch Metrics from accumulated CM
        miou, acc, prec, recall, f1 = calculate_epoch_metrics(epoch_cm)
        
        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(miou)
        history['val_acc'].append(acc)
        history['val_prec'].append(prec)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        print(f"   📉 Loss (T/V): {avg_train_loss:.4f} / {avg_val_loss:.4f}")
        print(f"   📊 Val Metrics -> mIoU: {miou:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

        # --- CHECKPOINTING ---
        if f1 > best_metric: # Monitoring F1-Score
            print(f"   ⭐ New Best Model! (F1: {best_metric:.4f} -> {f1:.4f})")
            best_metric = f1
            patience_counter = 0
            best_confusion_matrix = epoch_cm.clone() # Save for final plot
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'dsunet_best.pth'))
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n⛔ Early stopping at epoch {epoch+1}")
                break
        
        torch.cuda.empty_cache()

    print("\n🏁 TRAINING FINISHED.")
    print("📊 Generating Visualizations...")
    
    # 1. Plot Metrics
    plot_comprehensive_metrics(history, PLOTS_DIR)
    
    # 2. Plot Confusion Matrix
    class_names = ["Background", "Main Lane", "Other Lane", "Turn Lane"] # Adjust names as needed
    plot_confusion_matrices(best_confusion_matrix, class_names, PLOTS_DIR)
    
    # 3. Export Deployable Model
    print("📦 Exporting Deployable Model...")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'dsunet_best.pth')))
    deploy_model = DeployableDSUnet(model, target_size=(IMG_HEIGHT, IMG_WIDTH)).to(DEVICE)
    deploy_model.eval()
    
    dummy_input = torch.randint(0, 255, (1, 360, 640, 3), dtype=torch.uint8).to(DEVICE)
    try:
        traced_model = torch.jit.trace(deploy_model, dummy_input)
        traced_model.save(os.path.join(MODELS_DIR, 'dsunet_deploy.pt'))
        print("✅ Export successful.")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        
    print(f"✅ All results saved in: {EXP_DIR}")

if __name__ == "__main__":
    main()