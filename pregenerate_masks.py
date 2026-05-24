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
    
    # Khởi tạo COCO API
    coco = COCO(json_path)
    image_ids = list(coco.imgs.keys())
    
    for img_id in tqdm(image_ids, desc="Pregenerating Masks"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # Tạo tên file mask tương ứng (.png)
        mask_filename = os.path.splitext(file_name)[0] + "_mask.png"
        mask_save_path = os.path.join(output_masks_dir, mask_filename)
        
        # Đảm bảo thư mục con tồn tại (nếu có cấu trúc thư mục con)
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        
        # Nếu đã tồn tại mặt nạ thì bỏ qua
        if os.path.exists(mask_save_path):
            continue
            
        # Lấy danh sách annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Tạo mặt nạ trống (background là 0)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Sắp xếp annotations để vẽ các vật thể nhỏ đè lên trên
        anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
        
        for ann in anns:
            cat_id = ann['category_id']
            pixel_mask = coco.annToMask(ann)
            mask[pixel_mask == 1] = cat_id
            
        # Resize mặt nạ về kích thước chuẩn 256x512 bằng phép nội suy NEAREST
        mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Lưu mặt nạ dưới dạng ảnh PNG 8-bit đơn kênh
        Image.fromarray(mask_resized).save(mask_save_path)

if __name__ == "__main__":
    # 1. Tiền giải nén tập huấn luyện (Train)
    pregenerate_dataset_masks(
        json_path="data/train/_annotations.coco.json",
        images_dir="data/train",
        output_masks_dir="data/train_masks"
    )
    
    # 2. Tiền giải nén tập xác thực (Valid)
    pregenerate_dataset_masks(
        json_path="data/valid/_annotations.coco.json",
        images_dir="data/valid",
        output_masks_dir="data/valid_masks"
    )
    
    print("\n[SUCCESS] Pregenerated all dataset masks successfully!")
