import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from tqdm import tqdm
import random

# --- 0. SYSTEM CONFIGURATION ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 

# --- 1. HYPERPARAMETERS ---
# IMPORTANT: Adjust this based on your dataset labels
# 0 = Background, 1 = Lane A, 2 = Lane B (Total 3 classes)
NUM_CLASSES = 3 

EXPERIMENT_NAME = f"DSUnet_MultiClass{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_OUTPUT_DIR = r'E:\DSUnet-drivable_area_segmentation\Experiments'
EXP_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME)

MODELS_DIR = os.path.join(EXP_DIR, 'models')
LOGS_DIR = os.path.join(EXP_DIR, 'logs')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')

for d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Experiment: {EXPERIMENT_NAME}")
print(f"✅ Device: {DEVICE} | Classes: {NUM_CLASSES}")

# PATHS
DATASET_ROOT = r'E:\DSUnet-drivable_area_segmentation\Data'
SOURCE_IMG_DIR = os.path.join(DATASET_ROOT, 'train', 'images')
SOURCE_MASK_DIR = os.path.join(DATASET_ROOT, 'train', 'masks')

# Model Parameters
IMG_HEIGHT = 288 
IMG_WIDTH = 512
BATCH_SIZE = 8
EPOCHS = 150         
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

        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) 
        
        mask = torch.from_numpy(mask).long()
        img = torch.from_numpy(img)

        return img, mask, img_path

# --- 5. UTILS ---
def compute_metrics(preds, targets, num_classes):
    pred_labels = torch.argmax(preds, dim=1)
    
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred_labels == cls)
        target_inds = (targets == cls)
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            continue
        else:
            iou_list.append(intersection / union)
            
    miou = torch.tensor(iou_list).mean().item() if iou_list else 0.0
    
    correct = (pred_labels == targets).sum().float()
    total = torch.numel(targets)
    acc = (correct / total).item()
    
    return miou, acc

def plot_history(history, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_chart.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['val_miou'], label='Val mIoU', color='green')
    plt.title('Validation Mean IoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'miou_chart.png'))
    plt.close()

def sanity_check_dataset(img_paths, mask_paths):
    print("\n🔍 SANITY CHECK (Checking data quality)...")
    indices = np.random.choice(len(mask_paths), min(5, len(mask_paths)), replace=False)
    
    all_values = set()
    for idx in indices:
        mask = cv2.imread(mask_paths[idx], 0)
        unique_vals = np.unique(mask)
        all_values.update(unique_vals)
        print(f"   - File {os.path.basename(mask_paths[idx])}: Values found {unique_vals}")
    
    print(f"✅ Pixel values found in sample: {sorted(list(all_values))}")
    if max(all_values) >= NUM_CLASSES:
        print(f"⚠️ WARNING: Found pixel value {max(all_values)} >= NUM_CLASSES ({NUM_CLASSES}).")
        print("   Please ensure NUM_CLASSES is set correctly in configuration!")
    else:
        print("✅ Data appears valid.")

# --- 6. MAIN (FIXED PATH MATCHING) ---
def main():
    def get_paths(img_dir, mask_dir):
        """
        Robustly matches image and mask files even if names differ slightly.
        Common patterns: 
        - Image: abc.jpg -> Mask: abc.png
        - Image: abc.jpg -> Mask: abc_mask.png
        - Image: abc.jpg -> Mask: abc_drivable_id.png
        """
        if not os.path.exists(img_dir): 
            print(f"❌ Image directory not found: {img_dir}")
            return [], []
        if not os.path.exists(mask_dir): 
            print(f"❌ Mask directory not found: {mask_dir}")
            return [], []

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        pairs_img, pairs_mask = [], []
        
        print(f"🔍 Found {len(img_paths)} images. Attempting to match masks...")

        for img_p in img_paths:
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            
            # Potential mask filenames to check
            candidates = [
                f"{base_name}.png",                 # Exact match
                f"{base_name}_drivable_id.png",     # BDD100K style
                f"{base_name}_mask.png",            # Common style
                f"{base_name}_lane.png"             # Lane style
            ]
            
            found = False
            for cand in candidates:
                mask_p = os.path.join(mask_dir, cand)
                if os.path.exists(mask_p):
                    pairs_img.append(img_p)
                    pairs_mask.append(mask_p)
                    found = True
                    break
            
            # Uncomment below line to debug missing files
            # if not found: print(f"   ⚠️ Mask not found for: {base_name}")

        print(f"✅ Successfully matched {len(pairs_img)} image-mask pairs.")
        return pairs_img, pairs_mask

    # 1. Load Paths
    all_imgs, all_masks = get_paths(SOURCE_IMG_DIR, SOURCE_MASK_DIR)
    if not all_imgs: 
        print("❌ Dataset not found! Check SOURCE_IMG_DIR and SOURCE_MASK_DIR paths.")
        return

    # 2. Sanity Check
    sanity_check_dataset(all_imgs, all_masks)

    # 3. Split Data
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(all_imgs, all_masks, test_size=0.2, random_state=42)
    print(f"🔹 Training samples: {len(train_imgs)} | Validation samples: {len(val_imgs)}")

    # 4. DataLoader
    train_ds = MultiClassLaneDataset(train_imgs, train_masks, (IMG_HEIGHT, IMG_WIDTH), augment=True)
    val_ds = MultiClassLaneDataset(val_imgs, val_masks, (IMG_HEIGHT, IMG_WIDTH), augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 5. Init Model & Loss
    model = DSUnet(n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() 
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    best_metrics = {'miou': 0.0, 'epoch': 0, 'loss': 0}
    patience_counter = 0 
    
    print("\n🔥 STARTING MULTI-CLASS TRAINING LOOP...")
    
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
        total_acc = 0
        
        with torch.no_grad():
            for imgs, masks, paths in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                with autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                
                val_loss_accum += loss.item()
                miou, acc = compute_metrics(logits, masks, NUM_CLASSES)
                total_miou += miou
                total_acc += acc

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)
        avg_miou = total_miou / len(val_loader)
        avg_acc = total_acc / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(avg_miou)

        print(f"   📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   📊 Val mIoU: {avg_miou:.4f} | Pixel Acc: {avg_acc:.4f}")

        # Check Early Stopping
        if avg_miou > best_metrics['miou']:
            print(f"   ⭐ NEW BEST MODEL! (mIoU: {best_metrics['miou']:.4f} -> {avg_miou:.4f})")
            patience_counter = 0 
            best_metrics = {'miou': avg_miou, 'epoch': epoch+1, 'loss': avg_val_loss}
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'dsunet_multiclass_best.pth'))
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n⛔ EARLY STOPPING AT EPOCH {epoch+1}")
                break

        torch.cuda.empty_cache()

    print("\n🏁 TRAINING FINISHED.")
    plot_history(history, PLOTS_DIR)
    
    # --- EXPORT DEPLOYABLE MODEL ---
    print("\n📦 EXPORTING DEPLOYABLE MODEL...")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'dsunet_multiclass_best.pth')))
    
    deploy_model = DeployableDSUnet(model, target_size=(IMG_HEIGHT, IMG_WIDTH)).to(DEVICE)
    deploy_model.eval()
    
    dummy_input = torch.randint(0, 255, (1, 360, 640, 3), dtype=torch.uint8).to(DEVICE)
    
    try:
        traced_model = torch.jit.trace(deploy_model, dummy_input)
        save_name = f"dsunet_deploy_miou_{best_metrics['miou']:.3f}.pt"
        traced_model.save(os.path.join(MODELS_DIR, save_name))
        print(f"✅ EXPORT SUCCESS: {save_name}")
    except Exception as e:
        print(f"❌ EXPORT FAILED: {e}")

if __name__ == "__main__":
    main()