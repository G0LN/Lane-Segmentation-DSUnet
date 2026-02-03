import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix, f1_score

# --- 0. SYSTEM CONFIGURATION ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# --- 1. HYPERPARAMETERS ---
# 0 = Background, 1 = Lane A, 2 = Lane B (Total 3 classes)
NUM_CLASSES = 4

EXPERIMENT_NAME = f"DSUnet{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ĐƯỜNG DẪN CẦN CHỈNH LẠI THEO MÁY CỦA BẠN
BASE_OUTPUT_DIR = r'E:\DSUnet-drivable_area_segmentation\Experiments' 
DATASET_ROOT = r'E:\DSUnet-drivable_area_segmentation\Data\Lane-Segmentation-Auto.v1i.png-mask-semantic'

EXP_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME)
MODELS_DIR = os.path.join(EXP_DIR, 'models')
LOGS_DIR = os.path.join(EXP_DIR, 'logs')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')

for d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Experiment: {EXPERIMENT_NAME}")
print(f"✅ Device: {DEVICE} | Classes: {NUM_CLASSES}")

# PATHS CONFIG (Cấu trúc mới: Train và Valid riêng biệt, bên trong chứa cả ảnh lẫn mask)
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VALID_DIR = os.path.join(DATASET_ROOT, 'valid')

# Model Parameters
IMG_HEIGHT = 288
IMG_WIDTH = 512
BATCH_SIZE = 2
EPOCHS = 200
LEARNING_RATE = 1e-4
PATIENCE = 20

# --- 2. DEPLOYABLE MODEL (Wrapper) ---
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
        return torch.softmax(logits, dim=1)

# --- 3. CORE MODEL (DSUnet) ---
# --- ĐÃ CHỈNH SỬA CẤU TRÚC ĐỂ ĐÚNG LOGIC DROPOUT/SKIP CONNECTION ---
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
    def __init__(self, n_classes=NUM_CLASSES):
        super(DSUnet, self).__init__()
        # Encoder 1
        self.c1 = StandardBlock(3, 64)
        self.p1 = nn.MaxPool2d(2)
        
        # Encoder 2
        self.c2 = DSUnetBlock(64, 128)
        self.p2 = nn.MaxPool2d(2)
        
        # Encoder 3
        self.c3 = DSUnetBlock(128, 256)
        self.p3 = nn.MaxPool2d(2)
        
        # Encoder 4 (Layer 512)
        self.c4 = DSUnetBlock(256, 512)
        self.drop4 = nn.Dropout(0.25) # Dropout này dành cho nhánh Skip Connection
        self.p4 = nn.MaxPool2d(2)
        
        # Bottleneck (Layer 1024)
        self.b = DSUnetBlock(512, 1024)
        self.drop_b = nn.Dropout(0.25) # Dropout trước khi upsample
        
        # Decoder
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
        # --- ENCODER ---
        c1 = self.c1(x)             # 64
        c2 = self.c2(self.p1(c1))   # 128
        c3 = self.c3(self.p2(c2))   # 256
        
        # --- SPLIT FLOW AT LAYER 512 ---
        feat4 = self.c4(self.p3(c3)) # Feature map gốc của lớp 512
        
        # Nhánh 1: Đi xuống (Pooling) -> Không qua Dropout
        p4 = self.p4(feat4) 
        
        # Nhánh 2: Đi sang ngang (Skip Connection) -> Qua Dropout
        c4_skip = self.drop4(feat4)
        
        # --- BOTTLENECK ---
        # Dropout ngay sau khối Bottleneck 1024
        b = self.drop_b(self.b(p4))
        
        # --- DECODER ---
        # Concat với c4_skip (đã qua dropout)
        u6 = self.up6(b)
        # Fix size mismatch if any
        if u6.size() != c4_skip.size():
            u6 = torch.nn.functional.interpolate(u6, size=c4_skip.shape[2:], mode='bilinear', align_corners=False)
            
        u6 = self.c6(torch.cat([u6, c4_skip], dim=1))

        u7 = self.up7(u6)
        if u7.size() != c3.size():
            u7 = torch.nn.functional.interpolate(u7, size=c3.shape[2:], mode='bilinear', align_corners=False)
        u7 = self.c7(torch.cat([u7, c3], dim=1))
        
        u8 = self.up8(u7)
        if u8.size() != c2.size():
            u8 = torch.nn.functional.interpolate(u8, size=c2.shape[2:], mode='bilinear', align_corners=False)
        u8 = self.c8(torch.cat([u8, c2], dim=1))
        
        u9 = self.up9(u8)
        if u9.size() != c1.size():
            u9 = torch.nn.functional.interpolate(u9, size=c1.shape[2:], mode='bilinear', align_corners=False)
        u9 = self.c9(torch.cat([u9, c1], dim=1))
        
        return self.out(u9)

# --- 4. DATASET ---
class MultiClassLaneDataset(Dataset):
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
        mask = cv2.imread(mask_path, 0) # Read as grayscale

        # Resize
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

        # Normalize & Tensor
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) 
        
        mask = torch.from_numpy(mask).long()
        img = torch.from_numpy(img)

        return img, mask, img_path

# --- 5. METRICS & PLOTTING UTILS ---
def compute_metrics_batch(preds, targets, num_classes):
    """
    Tính toán metrics cho một batch.
    Trả về: (miou, acc, f1, confusion_matrix_batch)
    """
    pred_labels = torch.argmax(preds, dim=1) # [B, H, W]
    
    # --- Confusion Matrix Calculation (GPU Optimized) ---
    # Flatten tensors
    p = pred_labels.view(-1)
    t = targets.view(-1)
    
    # Chỉ tính pixel hợp lệ (nếu mask có giá trị 255 ignore thì lọc ở đây, dataset này giả sử sạch)
    mask = (t >= 0) & (t < num_classes)
    p = p[mask]
    t = t[mask]
    
    # Trick tính bincount để ra confusion matrix nhanh trên GPU
    # index = class_id_target * num_classes + class_id_pred
    indices = t * num_classes + p
    count = torch.bincount(indices, minlength=num_classes**2)
    cm_batch = count.reshape(num_classes, num_classes) # [Num_classes, Num_classes]
    
    # --- Metrics Calculation từ CM batch ---
    # Intersection = diag
    intersection = torch.diag(cm_batch)
    # Union = row_sum + col_sum - diag
    row_sum = cm_batch.sum(dim=1) # support (targets)
    col_sum = cm_batch.sum(dim=0) # predictions
    union = row_sum + col_sum - intersection
    
    iou = intersection / (union + 1e-6)
    miou = iou.mean().item()
    
    total_pixels = row_sum.sum().item()
    correct_pixels = intersection.sum().item()
    acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # F1 Score (Dice Coefficient per class) = 2*TP / (2*TP + FP + FN)
    # 2 * Intersection / (row_sum + col_sum)
    f1_per_class = 2 * intersection / (row_sum + col_sum + 1e-6)
    f1_macro = f1_per_class.mean().item()

    return miou, acc, f1_macro, cm_batch

def plot_history_extended(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss Chart
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='.')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_convergence.png'))
    plt.close()

    # 2. Metrics Chart (mIoU & F1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_miou'], label='Val mIoU', color='green', marker='.')
    plt.plot(epochs, history['val_f1'], label='Val F1-Score', color='blue', marker='.')
    plt.title('Validation Metrics (mIoU & F1)')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_chart.png'))
    plt.close()

def plot_confusion_matrix_final(cm_tensor, classes, save_dir):
    """
    Vẽ Confusion Matrix dạng Normal và Percent
    """
    cm = cm_tensor.cpu().numpy()
    
    # 1. Normal Count CM
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Pixel Counts)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_count.png'))
    plt.close()
    
    # 2. Percentage CM (Normalized by Row/True Label)
    # Tránh chia cho 0
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Normalized %)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_percent.png'))
    plt.close()

# --- 6. LOGIC LOAD DATA MỚI ---
def get_mixed_dataset_paths(dir_path):
    """
    Quét thư mục chứa cả ảnh và mask.
    Quy tắc match:
    - Ảnh: abc.jpg
    - Mask: abc_mask.png
    """
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        return [], []
    
    # Lấy tất cả file jpg
    all_files = os.listdir(dir_path)
    img_files = [f for f in all_files if f.lower().endswith('.jpg')]
    
    img_paths = []
    mask_paths = []
    
    print(f"🔍 Scanning {dir_path}...")
    
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0] # Bỏ đuôi .jpg
        
        # Tạo tên mask kỳ vọng
        expected_mask_name = f"{base_name}_mask.png"
        
        img_full_path = os.path.join(dir_path, img_file)
        mask_full_path = os.path.join(dir_path, expected_mask_name)
        
        if os.path.exists(mask_full_path):
            img_paths.append(img_full_path)
            mask_paths.append(mask_full_path)
        
    print(f"✅ Found {len(img_paths)} pairs in {os.path.basename(dir_path)}")
    return img_paths, mask_paths

# --- 7. MAIN FUNCTION ---
def main():
    # 1. Load Paths (Cấu trúc mới)
    print("🔄 Loading datasets from Train/Valid folders (Mixed content)...")
    train_imgs, train_masks = get_mixed_dataset_paths(TRAIN_DIR)
    val_imgs, val_masks = get_mixed_dataset_paths(VALID_DIR)
    
    if not train_imgs or not val_imgs:
        print("❌ Dataset Error: Could not find paired images/masks in Train or Valid folder.")
        print("   Please check filenames: 'name.jpg' needs 'name_mask.png'")
        return

    # 2. DataLoader
    train_ds = MultiClassLaneDataset(train_imgs, train_masks, (IMG_HEIGHT, IMG_WIDTH), augment=True)
    val_ds = MultiClassLaneDataset(val_imgs, val_masks, (IMG_HEIGHT, IMG_WIDTH), augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Setup Training
    model = DSUnet(n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # History tracker
    history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_f1': []}
    best_metrics = {'miou': 0.0, 'epoch': 0}
    patience_counter = 0
    
    # Global CM (Tổng kết cuối cùng của validation set tốt nhất)
    best_confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long).to(DEVICE)
    
    print("\n🔥 STARTING TRAINING LOOP...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VALIDATE ---
        model.eval()
        val_loss_accum = 0
        total_miou = 0
        total_f1 = 0
        
        # CM tạm thời cho epoch này
        current_epoch_cm = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                with autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                
                val_loss_accum += loss.item()
                
                # Tính metrics và tích lũy CM
                miou, acc, f1, cm_batch = compute_metrics_batch(logits, masks, NUM_CLASSES)
                total_miou += miou
                total_f1 += f1
                current_epoch_cm += cm_batch

        # Stats calculation
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)
        avg_miou = total_miou / len(val_loader)
        avg_f1 = total_f1 / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(avg_miou)
        history['val_f1'].append(avg_f1)

        print(f"   📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   📊 Val mIoU: {avg_miou:.4f} | F1-Score: {avg_f1:.4f}")

        # Check Early Stopping & Save Best
        if avg_miou > best_metrics['miou']:
            print(f"   ⭐ NEW BEST MODEL! (mIoU: {best_metrics['miou']:.4f} -> {avg_miou:.4f})")
            best_metrics = {'miou': avg_miou, 'epoch': epoch+1}
            patience_counter = 0
            
            # Lưu model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'dsunet_best.pth'))
            
            # Lưu lại CM của model tốt nhất để vẽ sau này
            best_confusion_matrix = current_epoch_cm.clone()
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n⛔ EARLY STOPPING AT EPOCH {epoch+1}")
                break
        
        torch.cuda.empty_cache()

    print("\n🏁 TRAINING FINISHED.")
    
    # 4. PLOTTING RESULTS
    print("📊 Generating plots...")
    plot_history_extended(history, PLOTS_DIR)
    
    # Vẽ Confusion Matrix từ model tốt nhất
    # class_names = [f"Class {i}" for i in range(NUM_CLASSES)] # Bạn có thể đổi thành ['Background', 'Lane', 'Drivable'] tùy ý
    class_names = ["Background", "Main lane", "Other lane", "Turn lane"]  # Cập nhật tên lớp phù hợp với NUM_CLASSES = 4
    plot_confusion_matrix_final(best_confusion_matrix, class_names, PLOTS_DIR)
    
    print(f"✅ All results saved in: {EXP_DIR}")

    # 5. EXPORT
    print("\n📦 EXPORTING DEPLOYABLE MODEL...")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'dsunet_best.pth')))
    deploy_model = DeployableDSUnet(model, target_size=(IMG_HEIGHT, IMG_WIDTH)).to(DEVICE)
    deploy_model.eval()
    
    dummy_input = torch.randint(0, 255, (1, 360, 640, 3), dtype=torch.uint8).to(DEVICE)
    try:
        traced_model = torch.jit.trace(deploy_model, dummy_input)
        traced_model.save(os.path.join(MODELS_DIR, 'dsunet_deploy.pt'))
        print("✅ Export success.")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()