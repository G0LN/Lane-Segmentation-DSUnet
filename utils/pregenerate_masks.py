import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image

def pregenerate_dataset_masks(json_path, images_dir, output_masks_dir, img_height=256, img_width=512):
    print("=" * 60)
    print("STARTING MASK PREGENERATION PROCESS")
    print(f"-> Annotations: {json_path}")
    print(f"-> Output Dir: {output_masks_dir}")
    print("=" * 60)
    
    if not os.path.exists(json_path):
        print(f"Error: Annotations file not found at {json_path}")
        return
        
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Initialize COCO API
    coco = COCO(json_path)
    image_ids = list(coco.imgs.keys())
    
    for img_id in tqdm(image_ids, desc="Pregenerating Masks"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # Create corresponding mask filename (.png)
        mask_filename = os.path.splitext(file_name)[0] + "_mask.png"
        mask_save_path = os.path.join(output_masks_dir, mask_filename)
        
        # Ensure subdirectory exists
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        
        # Skip if already exists
        if os.path.exists(mask_save_path):
            continue
            
        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Blank mask (background = 0)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Sort annotations by area descending so smaller objects are drawn on top
        anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
        
        for ann in anns:
            cat_id = ann['category_id']
            pixel_mask = coco.annToMask(ann)
            mask[pixel_mask == 1] = cat_id
            
        # Resize to standard height/width using NEAREST interpolation to preserve class labels
        mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Save as 8-bit single-channel PNG
        Image.fromarray(mask_resized).save(mask_save_path)

if __name__ == "__main__":
    # If run directly as a utility script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pregenerate_dataset_masks(
        json_path="data/train/_annotations.coco.json",
        images_dir="data/train",
        output_masks_dir="data/train_masks"
    )
    pregenerate_dataset_masks(
        json_path="data/valid/_annotations.coco.json",
        images_dir="data/valid",
        output_masks_dir="data/valid_masks"
    )
    print("\n[SUCCESS] Pregenerated all dataset masks successfully!")
