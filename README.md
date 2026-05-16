# DSUnet Lane Segmentation Training Project

Dự án này cung cấp cấu trúc module hóa chuẩn cho việc huấn luyện mô hình phân đoạn đa làn đường (và các đối tượng khác như xe, vật cản) sử dụng kiến trúc DSUnet.

## Cấu trúc thư mục
- `configs/`: File YAML chứa siêu tham số.
- `data/`: Xử lý Dataset, Dataloader và Augmentations.
- `models/`: Định nghĩa kiến trúc DSUnet.
- `utils/`: Metrics (mIoU), Loss (DiceLoss, CrossEntropy), Logger.
- `train.py`: Vòng lặp huấn luyện chính.
- `evaluate.py`: Đánh giá mô hình.
- `inference.py`: Dự đoán trên ảnh thực tế.

## Hướng dẫn sử dụng

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chuẩn bị dữ liệu:**
   Chỉnh sửa đường dẫn trong `configs/default.yaml` để trỏ tới dataset của bạn.

3. **Huấn luyện mô hình:**
   ```bash
   python train.py
   ```

4. **Đánh giá mô hình:**
   Mở file `evaluate.py` và bỏ comment phần example usage ở cuối file, sau đó chạy:
   ```bash
   python evaluate.py
   ```

5. **Dự đoán:**
   Mở file `inference.py` và điều chỉnh các tham số đầu vào.
   ```bash
   python inference.py
   ```
