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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Load mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create empty mask (background is 0)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Sort annotations by area descending so smaller objects are drawn on top
        anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
        
        for ann in anns:
            cat_id = ann['category_id']
            # annToMask returns 1 for pixels inside polygon
            pixel_mask = self.coco.annToMask(ann)
            mask[pixel_mask == 1] = cat_id

        # Resize image and mask
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        # Use NEAREST for mask to avoid interpolating category IDs
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
