import torch
import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# --- 1. CẤU HÌNH (CONFIGURATION) ---
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_MultiClass_Fixed_20251218_150629\models\dsunet_deploy_miou_0.679.pt"
INPUT_PATH = r"E:\DSUnet-drivable_area_segmentation\Data\Video\mixkit-point-of-view-from-a-bus-passenger-seat-roading-in-4394-hd-ready.mp4" 
OUTPUT_DIR = "inference_results_optimized"

# Cấu hình xử lý
MIN_LANE_AREA = 100     # Diện tích tối thiểu (pixel) trên ảnh nhỏ để chấp nhận là 1 làn đường
SAMPLE_STEP = 10        # Bước nhảy khi quét hàng (trên ảnh nhỏ)

# Màu sắc
OVERLAY_COLORS = {
    0: [0, 0, 0],
    1: [0, 255, 0],     # Lane A (Green)
    2: [0, 0, 255],     # Lane B (Red)
}

PATH_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 255],   # Path A (Magenta)
    2: [0, 255, 255],   # Path B (Yellow)
}

ALPHA = 0.5

# --- 2. CLASS DỰ ĐOÁN ---
class LanePredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔄 Loading model: {model_path}")
        print(f"⚙  Device: {self.device}")
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            # Warm-up
            dummy = torch.zeros((1, 288, 512, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad(): self.model(dummy)
            print("✅ Model Ready!")
        except Exception as e:
            print(f"❌ Error: {e}")
            exit()

    def predict_raw(self, img_bgr):
        """
        Trả về mask thô kích thước nhỏ (theo output model) để tối ưu tốc độ xử lý.
        """
        img_tensor = torch.from_numpy(img_bgr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model(img_tensor)
        # Output: [1, C, H, W] -> [H, W]
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return pred_mask

# --- 3. XỬ LÝ PATH (OPTIMIZED) ---
def get_separated_paths(binary_mask, cls_id, scale_x, scale_y):
    """
    1. Tách các làn đường riêng biệt (Connected Components).
    2. Tính Cubic Path cho từng làn.
    3. Scale tọa độ về kích thước video gốc.
    """
    # Bước 1: Tiền xử lý - Loại bỏ nhiễu (Outlier Removal)
    # Dùng phép mở (Opening) để xóa các điểm nhiễu lốm đốm
    kernel = np.ones((3,3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Bước 2: Tách các đối tượng (Instance Separation)
    # Đây là bước quan trọng để không nối 2 làn đường xa nhau
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    
    paths = []
    
    # Duyệt qua từng "hòn đảo" (blob) tìm thấy (bỏ qua label 0 là nền đen)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Bỏ qua nếu vùng quá nhỏ (nhiễu)
        if area < MIN_LANE_AREA: 
            continue
            
        # Lấy bounding box để quét cho nhanh (Optimization)
        x_box, y_box, w_box, h_box = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Bước 3: Sampling (Lấy mẫu điểm)
        sample_x = []
        sample_y = []
        
        # Chỉ quét trong vùng bounding box của blob này
        # Quét từ dưới lên trên
        for r in range(y_box + h_box - 1, y_box, -SAMPLE_STEP):
            # Cắt một lát ngang trong vùng labels
            row_slice = labels[r, x_box : x_box + w_box]
            
            # Tìm các pixel thuộc về blob thứ i
            indices = np.where(row_slice == i)[0]
            
            if len(indices) > 0:
                # Tính tâm tương đối + offset x_box
                center_x = x_box + int((indices[0] + indices[-1]) / 2)
                sample_x.append(center_x)
                sample_y.append(r)
        
        # Cần ít nhất 4 điểm để fit bậc 3
        if len(sample_y) < 4: 
            continue

        try:
            # Bước 4: Fit Đa thức Bậc 3
            fit_params = np.polyfit(sample_y, sample_x, 3)
            poly_func = np.poly1d(fit_params)
            
            # Tạo điểm vẽ trơn tru
            plot_y = np.linspace(min(sample_y), max(sample_y), num=50)
            plot_x = poly_func(plot_y)
            
            # Bước 5: Scale toạ độ về video gốc (Upscaling)
            plot_x_scaled = plot_x * scale_x
            plot_y_scaled = plot_y * scale_y
            
            # Gom lại thành format polylines
            pts = np.array([np.transpose(np.vstack([plot_x_scaled, plot_y_scaled]))], np.int32)
            paths.append(pts)
            
        except:
            continue
            
    return paths

# --- 4. VISUALIZATION ---
def draw_results(frame, mask_small, paths_dict):
    """
    Vẽ overlay và path lên khung hình gốc.
    """
    h_orig, w_orig = frame.shape[:2]
    
    # 1. Vẽ Overlay (Resize mask nhỏ -> to)
    # Dùng Nearest để giữ nguyên giá trị class
    mask_large = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    color_mask = np.zeros_like(frame)
    unique_ids = np.unique(mask_small) # Check trên mask nhỏ cho nhanh
    
    for cls_id in unique_ids:
        if cls_id == 0: continue
        if cls_id in OVERLAY_COLORS:
            color_mask[mask_large == cls_id] = OVERLAY_COLORS[cls_id]
            
    # Blend Overlay
    mask_bool = (mask_large > 0)
    if np.any(mask_bool):
        # Chỉ blend vùng cần thiết để tăng tốc
        frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1-ALPHA, color_mask[mask_bool], ALPHA, 0)
        
    # 2. Vẽ Path (Đã được tính toán riêng biệt)
    for cls_id, paths in paths_dict.items():
        color = PATH_COLORS.get(cls_id, [255, 255, 255])
        for line_pts in paths:
            cv2.polylines(frame, line_pts, isClosed=False, color=color, thickness=4)
            
    return frame

# --- 5. MAIN PROCESS ---
def process_video(predictor, video_path):
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"optimized_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_input, (width, height))
    
    print(f"🎬 Processing: {width}x{height} @ {fps_input}fps")
    
    # Pre-calculate scale factors
    # Model output size is usually 512x288 based on your previous code
    # If your model output is different, change these values
    MODEL_W, MODEL_H = 512, 288 
    scale_x = width / MODEL_W
    scale_y = height / MODEL_H
    
    pbar = tqdm(total=total_frames)
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Resize Input cho Model (Tăng tốc Inference)
        img_small = cv2.resize(frame, (MODEL_W, MODEL_H))
        
        # 2. Predict (Trả về mask nhỏ)
        mask_small = predictor.predict_raw(img_small)
        
        # 3. Tính toán Path cho từng class (Trên không gian nhỏ)
        paths_dict = {}
        unique_classes = np.unique(mask_small)
        for cls_id in unique_classes:
            if cls_id == 0: continue
            
            # Tạo binary mask cho class hiện tại
            bin_mask = (mask_small == cls_id).astype(np.uint8) * 255
            
            # Tách làn và tính path riêng biệt
            paths = get_separated_paths(bin_mask, cls_id, scale_x, scale_y)
            paths_dict[cls_id] = paths
            
        # 4. Vẽ kết quả lên Frame gốc (High Quality Visualization)
        result = draw_results(frame, mask_small, paths_dict)
        
        # 5. FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        cv2.putText(result, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(result)
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()
    print(f"\n✅ Video saved: {save_path}")

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model path incorrect.")
        return
    
    predictor = LanePredictor(MODEL_PATH)
    process_video(predictor, INPUT_PATH)

if __name__ == "__main__":
    main()