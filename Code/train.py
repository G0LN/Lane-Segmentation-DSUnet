import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler # Giữ GradScaler
# autocast được dùng trực tiếp từ torch.amp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from tqdm import tqdm
import json
import gc

# --- 0. SYSTEM CONFIGURATION ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 

# --- 1. HYPERPARAMETERS & PATHS ---
EXPERIMENT_NAME = f"DSUnet_Deploy_Fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_OUTPUT_DIR = r'E:\DSUnet-drivable_area_segmentation\Experiments' # Sửa lại đúng đường dẫn của bạn
EXP_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME)

MODELS_DIR = os.path.join(EXP_DIR, 'models')
LOGS_DIR = os.path.join(EXP_DIR, 'logs')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')

for d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Experiment: {EXPERIMENT_NAME}")
print(f"✅ Device: {DEVICE}")

DATASET_ROOT = r'E:\DSUnet-drivable_area_segmentation\Data'
SOURCE_IMG_DIR = os.path.join(DATASET_ROOT, 'train', 'images')
SOURCE_MASK_DIR = os.path.join(DATASET_ROOT, 'train', 'masks')

# Model Parameters
IMG_HEIGHT = 288 
IMG_WIDTH = 512
BATCH_SIZE = 4
EPOCHS = 200         
LEARNING_RATE = 1e-4
PATIENCE = 25       

# --- 2. DEPLOYABLE MODEL WRAPPER ---
class DeployableDSUnet(nn.Module):
    def __init__(self, core_model, target_size=(288, 512)):
        super(DeployableDSUnet, self).__init__()
        self.core_model = core_model
        self.target_h = target_size[0]
        self.target_w = target_size[1]
        self.register_buffer('scale', torch.tensor(255.0))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.float() / self.scale
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = torch.nn.functional.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        logits = self.core_model(x)
        return torch.sigmoid(logits)

# --- 3. CORE MODEL ARCHITECTURE (DSUnet) ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DSUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSUnetBlock, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class StandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DSUnet(nn.Module):
    def __init__(self, n_classes=1):
        super(DSUnet, self).__init__()
        self.c1 = StandardBlock(3, 64)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = DSUnetBlock(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = DSUnetBlock(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = DSUnetBlock(256, 512)
        self.drop4 = nn.Dropout(0.25)
        self.p4 = nn.MaxPool2d(2)
        
        self.b = DSUnetBlock(512, 1024)
        self.drop_b = nn.Dropout(0.25)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c6 = DSUnetBlock(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c7 = DSUnetBlock(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c8 = DSUnetBlock(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c9 = DSUnetBlock(128, 64)
        
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.drop4(self.c4(self.p3(c3)))
        b = self.drop_b(self.b(self.p4(c4)))
        
        u6 = self.c6(torch.cat([self.up6(b), c4], dim=1))
        u7 = self.c7(torch.cat([self.up7(u6), c3], dim=1))
        u8 = self.c8(torch.cat([self.up8(u7), c2], dim=1))
        u9 = self.c9(torch.cat([self.up9(u8), c1], dim=1))
        
        return self.out(u9)

# --- 4. DATASET ---
class BDD100kDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=(288, 512), augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) 
        
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img), torch.from_numpy(mask), img_path

# --- 5. UTILS ---
def calculate_cm_components(preds, targets, threshold=0.5):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    tp = (preds_bin * targets).sum().item()
    tn = ((1 - preds_bin) * (1 - targets)).sum().item()
    fp = (preds_bin * (1 - targets)).sum().item()
    fn = ((1 - preds_bin) * targets).sum().item()
    return tp, tn, fp, fn

def plot_history(history, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_chart.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['val_f1'], label='Val F1-Score', color='green')
    plt.title('Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1_chart.png'))
    plt.close()

def plot_confusion_matrix(cm, save_path):
    # FIX LỖI: Chuyển cm sang int để vẽ heatmap không bị lỗi định dạng 'd'
    cm_int = cm.astype(int) 
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Lane'], yticklabels=['Background', 'Lane'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Best Epoch)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_optimal_threshold(model, loader, device):
    print("\n🔍 Searching for Optimal F1 Threshold...")
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Collecting predictions"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            y_true.append(masks.cpu().numpy().flatten())
            y_scores.append(probs.cpu().numpy().flatten())
    
    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"✅ Optimal Threshold: {best_thresh:.4f} (Max F1: {best_f1:.4f})")
    return best_thresh, best_f1

# --- 6. MAIN ---
def main():
    def get_paths(img_dir, mask_dir):
        if not os.path.exists(img_dir): return [], []
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        pairs_img, pairs_mask = [], []
        for img_p in img_paths:
            fname = os.path.splitext(os.path.basename(img_p))[0]
            mask_p = os.path.join(mask_dir, f"{fname}_drivable_id.png")
            if os.path.exists(mask_p):
                pairs_img.append(img_p)
                pairs_mask.append(mask_p)
        return pairs_img, pairs_mask

    all_imgs, all_masks = get_paths(SOURCE_IMG_DIR, SOURCE_MASK_DIR)
    if not all_imgs: print("❌ Dataset not found!"); return

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(all_imgs, all_masks, test_size=0.2, random_state=42)
    print(f"🔹 Training samples: {len(train_imgs)} | Validation samples: {len(val_imgs)}")

    train_ds = BDD100kDataset(train_imgs, train_masks, (IMG_HEIGHT, IMG_WIDTH), augment=True)
    val_ds = BDD100kDataset(val_imgs, val_masks, (IMG_HEIGHT, IMG_WIDTH), augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = DSUnet(n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_metrics = {'f1': 0.0, 'epoch': 0, 'acc': 0, 'prec': 0, 'rec': 0, 'loss': 0}
    patience_counter = 0 
    
    print("\n🔥 STARTING TRAINING LOOP...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            
            # [FIX] Sử dụng cú pháp mới torch.amp.autocast
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VALIDATE ---
        model.eval()
        val_loss_accum = 0
        epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0
        hard_examples = []

        with torch.no_grad():
            for imgs, masks, paths in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                # [FIX] Sử dụng cú pháp mới torch.amp.autocast
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                
                val_loss_accum += loss.item()
                tp, tn, fp, fn = calculate_cm_components(preds, masks)
                epoch_tp += tp; epoch_tn += tn; epoch_fp += fp; epoch_fn += fn
                
                # Hard Mining
                batch_iou = tp / (tp + fp + fn + 1e-6)
                if batch_iou < 0.5:
                    for p in paths: hard_examples.append((batch_iou, os.path.basename(p)))

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)
        
        epsilon = 1e-7
        precision = epoch_tp / (epoch_tp + epoch_fp + epsilon)
        recall = epoch_tp / (epoch_tp + epoch_fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        accuracy = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn + epsilon)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(f1)

        print(f"   📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   📊 Val F1: {f1:.4f} | Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        if f1 > best_metrics['f1']:
            print(f"   ⭐ NEW BEST MODEL! (F1: {best_metrics['f1']:.4f} -> {f1:.4f})")
            patience_counter = 0 
            best_metrics = {'f1': f1, 'epoch': epoch+1, 'acc': accuracy, 'prec': precision, 'rec': recall, 'loss': avg_val_loss}
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'dsunet_best_weights.pth'))
            
            # [FIXED] Cast sang int để tránh lỗi ValueError 'd' format
            best_cm = np.array([[epoch_tn, epoch_fp], [epoch_fn, epoch_tp]]).astype(int)
            plot_confusion_matrix(best_cm, os.path.join(PLOTS_DIR, 'best_confusion_matrix.png'))
            
            with open(os.path.join(LOGS_DIR, 'hard_examples.txt'), 'w') as f:
                hard_examples.sort(key=lambda x: x[0])
                for score, name in hard_examples[:50]: f.write(f"{name} (IoU ~ {score:.4f})\n")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Early Stopping Counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n⛔ EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                break

        torch.cuda.empty_cache()

    print("\n🏁 TRAINING FINISHED. STARTING POST-ANALYSIS...")
    plot_history(history, PLOTS_DIR)
    
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'dsunet_best_weights.pth')))
    best_thresh, best_f1_opt = find_optimal_threshold(model, val_loader, DEVICE)
    
    with open(os.path.join(LOGS_DIR, 'final_report.txt'), 'w') as f:
        f.write("=== BEST TRAINING RESULTS ===\n")
        f.write(f"Best Epoch: {best_metrics['epoch']}\n")
        f.write(f"F1-Score (Threshold 0.5): {best_metrics['f1']:.4f}\n")
        f.write(f"Optimal Threshold: {best_thresh:.4f}\n")
        f.write(f"Max Achievable F1: {best_f1_opt:.4f}\n")

    print("\n📦 EXPORTING DEPLOYABLE MODEL...")
    deploy_model = DeployableDSUnet(model, target_size=(IMG_HEIGHT, IMG_WIDTH)).to(DEVICE)
    deploy_model.eval()
    dummy_input = torch.randint(0, 255, (1, 360, 640, 3), dtype=torch.uint8).to(DEVICE)
    
    try:
        traced_model = torch.jit.trace(deploy_model, dummy_input)
        save_name = f"dsunet_deploy_f1_{best_f1_opt:.3f}.pt"
        traced_model.save(os.path.join(MODELS_DIR, save_name))
        print(f"✅ EXPORT SUCCESS: {save_name}")
    except Exception as e:
        print(f"❌ EXPORT FAILED: {e}")

if __name__ == "__main__":
    main()