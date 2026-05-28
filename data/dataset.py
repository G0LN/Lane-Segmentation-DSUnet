import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOLaneSegmentationDataset(Dataset):
    def __init__(self, images_dir, json_path, img_height, img_width, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            json_path (string): Path to the COCO JSON annotation file.
            img_height (int): Target height for resizing.
            img_width (int): Target width for resizing.
            transform (callable, optional): Optional transform to be applied.
        """
        self.images_dir = images_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        
        # Initialize COCO api for instance annotations
        print(f"Loading annotations from {json_path}...")
        self.coco = COCO(json_path)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Auto-pregenerate masks if missing or incomplete
        masks_dir = self.images_dir + "_masks"
        need_generate = False
        if not os.path.exists(masks_dir):
            need_generate = True
        else:
            try:
                mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
                if len(mask_files) < len(self.image_ids):
                    print(f"[Auto-Generator] Mask directory '{masks_dir}' is incomplete ({len(mask_files)}/{len(self.image_ids)}). Triggering regeneration...")
                    need_generate = True
            except Exception:
                need_generate = True
                
        if need_generate:
            print(f"[Auto-Generator] Missing masks detected. Automatically pregenerating mask PNGs from COCO JSON...")
            from utils.pregenerate_masks import pregenerate_dataset_masks
            pregenerate_dataset_masks(
                json_path=json_path,
                images_dir=images_dir,
                output_masks_dir=masks_dir,
                img_height=img_height,
                img_width=img_width
            )
            print(f"[Auto-Generator] Pregeneration completed for {images_dir}!\n")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # Load image
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Resize image
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        
        # Fast Path: Check if pre-generated PNG mask exists (e.g. under data/train_masks)
        masks_dir = self.images_dir + "_masks"
        mask_filename = os.path.splitext(file_name)[0] + "_mask.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if os.path.exists(mask_path):
            # Load the pre-generated and pre-resized 256x512 mask directly
            mask = Image.open(mask_path)
            mask = np.array(mask)
        else:
            # Fallback Path: Slow COCO polygon rendering on the fly
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Create empty mask (background is 0)
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            
            # Sort annotations by area descending so smaller objects are drawn on top
            anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
            
            for ann in anns:
                cat_id = ann['category_id']
                pixel_mask = self.coco.annToMask(ann)
                mask[pixel_mask == 1] = cat_id
                
            # Resize mask to target size (256x512)
            mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        # Apply transformations (e.g., albumentations)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Basic conversion to tensor if no transforms
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

# Maintain backward compatibility for import statements
LaneSegmentationDataset = COCOLaneSegmentationDataset
