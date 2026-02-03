import torch
import cv2
import numpy as np
import os
import time
import random
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. USER CONFIGURATION ---
# Path to the exported .pt model (TorchScript)
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_Deploy_Fixed_20251216_210448\models\dsunet_deploy_f1_0.879.pt"

# Input Path (Can be a specific image file OR a folder containing images)
# Example Folder: r"E:\DSUnet-drivable_area_segmentation\Data\test"
# Example File: r"E:\DSUnet-drivable_area_segmentation\Data\test\0a0a0b1a-7c39d841.jpg"
INPUT_PATH = r"E:\DSUnet-drivable_area_segmentation\Data\test\c1226eb9-27ecd843.jpg"

# Output directory for results
OUTPUT_ROOT = "inference_results_images"

# Threshold for determining lane pixels (Updated from final_report.txt)
THRESHOLD = 0.5189

# Number of random images to process if Input is a Folder
NUM_RANDOM_SAMPLES = 10

# --- 2. LANE DETECTOR CLASS ---
class LaneDetector:
    def __init__(self, model_path, threshold=0.5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        print(f"🔄 Loading model: {os.path.basename(model_path)}")
        try:
            # Load TorchScript model
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            
            # Warm-up the GPU
            dummy = torch.zeros((1, 360, 640, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad(): self.model(dummy)
            print("✅ Model is ready!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()

    def predict(self, img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        
        # Convert to Tensor [1, H, W, 3]
        img_tensor = torch.from_numpy(img_bgr).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            prob_map = self.model(img_tensor)
            
        # Post-processing
        prob_map = prob_map.squeeze().cpu().numpy()
        
        # Resize back to original size
        prob_map_resized = cv2.resize(prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        
        # Create Binary Mask (0 or 255) based on Threshold
        mask = (prob_map_resized > self.threshold).astype(np.uint8) * 255
        return mask

# --- 3. IMAGE PROCESSING UTILS ---
def create_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """Creates an overlay image by blending the mask color with the original image."""
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color # Green color (BGR)
    
    # Blend images
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return overlay

def process_and_display(detector, img_path, save_root):
    # 1. Read Image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠ Error reading file: {img_path}")
        return

    # 2. Predict
    mask = detector.predict(img_bgr)
    
    # 3. Create Overlay
    overlay = create_overlay(img_bgr, mask)
    
    # 4. Save results to separate folders
    filename = os.path.basename(img_path)
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Create sub-folders
    mask_dir = os.path.join(save_root, "saved_masks")
    overlay_dir = os.path.join(save_root, "saved_overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Save files
    cv2.imwrite(os.path.join(mask_dir, f"{filename_no_ext}_mask.png"), mask)
    cv2.imwrite(os.path.join(overlay_dir, f"{filename_no_ext}_overlay.jpg"), overlay)
    
    # 5. Display (using Matplotlib)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original: {filename}")
    plt.axis('off')
    
    # Binary Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Binary Mask (Thresh: {detector.threshold})")
    plt.axis('off')
    
    # Overlay Result
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_rgb)
    plt.title("Overlay Result")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() # Window will pop up, close it to process the next image

# --- 4. MAIN PROGRAM ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        exit()
        
    detector = LaneDetector(MODEL_PATH, threshold=THRESHOLD)
    
    # Handle Input Logic
    target_files = []
    
    if os.path.isfile(INPUT_PATH):
        print(f"🖼 Single Image Mode: {INPUT_PATH}")
        target_files.append(INPUT_PATH)
        
    elif os.path.isdir(INPUT_PATH):
        print(f"📂 Folder Mode: {INPUT_PATH}")
        # Find all images
        all_images = glob.glob(os.path.join(INPUT_PATH, "*.jpg")) + \
                     glob.glob(os.path.join(INPUT_PATH, "*.png"))
        
        if len(all_images) == 0:
            print("❌ Folder is empty or contains no images!")
            exit()
            
        # Select random samples
        num_take = min(NUM_RANDOM_SAMPLES, len(all_images))
        target_files = random.sample(all_images, num_take)
        print(f"🎲 Selected {num_take} random images for testing.")
        
    else:
        print(f"❌ Input path does not exist: {INPUT_PATH}")
        exit()

    # Start Processing Loop
    print("-" * 50)
    for fpath in tqdm(target_files, desc="Processing"):
        process_and_display(detector, fpath, OUTPUT_ROOT)
        
    print("-" * 50)
    print(f"✅ Results saved at: {os.path.abspath(OUTPUT_ROOT)}")
    print(f"   ├─ saved_masks/")
    print(f"   └─ saved_overlays/")