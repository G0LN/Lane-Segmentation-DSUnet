# Cấu trúc và Siêu tham số Mô hình DSUNet (Lane Detection)

Dựa trên các tài liệu kỹ thuật, dưới đây là chi tiết cấu trúc và thiết lập của mô hình DSUNet.

## 1. Kiến trúc Chi tiết (Architecture)

DSUNet là mạng CNN dạng **Encoder-Decoder** dựa trên UNet nhưng sử dụng **Tích chập tách sâu (Depthwise Separable Convolutions - DS)** để tối ưu hóa.

### Đặc điểm cấu trúc:
- **Đầu vào:** Hình ảnh RGB (3 kênh).
- **Đầu ra:** Ảnh nhị phân (1 kênh) xác định vị trí làn đường.
- **Tổng số lớp:** 40 lớp tích chập (Nhiều hơn UNet 23 lớp nhưng nhẹ hơn).
- **Tham số:** 6.01 triệu (UNet là 31.04 triệu).
- **Tốc độ:** 47 FPS.

### Thành phần chính:
1. **Khối Tích chập tách sâu (DS Block):** - 3x3 Depthwise Conv + BN + ReLU
   - 1x1 Pointwise Conv + BN + ReLU
   - Thay thế 17 lớp tích chập 3x3 tiêu chuẩn của UNet.
2. **Lấy mẫu (Sampling):**
   - **Downsampling:** Max Pooling 2x2.
   - **Upsampling:** Up-Conv 2x2 + ReLU.
3. **Kết nối (Skip Connections):** Sử dụng kỹ thuật Crop + Concat để giữ thông tin chi tiết.
4. **Điều chuẩn (Regularization):** 3 lớp Dropout để chống quá khớp.
5. **Lớp cuối:** 1x1 Standard Conv + Sigmoid.

## 2. Siêu tham số & Thiết lập Huấn luyện (Hyperparameters)

### Thiết lập cơ bản:
| Tham số | Giá trị |
| :--- | :--- |
| **Optimizer** | Adam |
| **Batch Size** | 1 |
| **Momentum** | 0.9 |
| **Total Epochs** | 100 |

### Chiến lược Tốc độ học (Learning Rate):
- **Epoch 1 - 75:** 10⁻⁴
- **Epoch 76 - 100:** 10⁻⁵

### Hàm mất mát (Loss Function):
Sử dụng **Weighted Cross-Entropy Loss** để giải quyết mất cân bằng lớp (tỷ lệ < 1/50).
- **Trọng số lớp dương (P):** $W_{pos} = N / (P + N)$
- **Trọng số lớp âm (N):** $W_{neg} = P / (P + N)$

---
*Ghi chú: DSUNet giúp duy trì độ chính xác của UNet trong khi giảm đáng kể tài nguyên tính toán.*
