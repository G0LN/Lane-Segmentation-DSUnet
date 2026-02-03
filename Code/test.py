import os
import cv2
import numpy as np

# Đường dẫn tới folder chứa ảnh mask
folder_path = r"E:\DSUnet-drivable_area_segmentation\Data\Lane-Segmentation-Auto.v2i.png-mask-semantic\valid"

count_images = 0  # số lượng ảnh có chứa pixel = 3

for filename in os.listdir(folder_path):
    if filename.endswith("_mask.png"):  # lọc đúng pattern tên file
        file_path = os.path.join(folder_path, filename)
        
        # đọc ảnh dưới dạng grayscale
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        # kiểm tra xem có pixel nào bằng 3 không
        if np.any(mask == 3):
            count_images += 1

print("Số lượng ảnh mask có pixel = 3 là:", count_images)
