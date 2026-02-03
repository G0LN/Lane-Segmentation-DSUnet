import torch
import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# --- 1. CẤU HÌNH (CONFIGURATION) ---
# Đảm bảo đường dẫn trỏ đúng tới file .pt bạn vừa export
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_test_20260122_134633\models\dsunet_deploy.pt" 
INPUT_VIDEO = r"E:\DSUnet-drivable_area_segmentation\Data\Video\mixkit-going-down-a-curved-highway-through-a-mountain-range-41576-hd-ready.mp4"
OUTPUT_DIR = r"E:\DSUnet-drivable_area_segmentation\Inference_Result_Waypoints4"

# Kích thước model (Khớp với lúc train/export)
MODEL_W, MODEL_H = 512, 288

# Màu sắc (BGR)
# 0: Background (Không vẽ), 1: Main Lane, 2: Other Lane, 3: Turn Lane
OVERLAY_COLORS = {
    1: [0, 255, 0],   # Green
    2: [0, 0, 255],   # Red
    3: [255, 0, 0]    # Blue
}

PATH_COLORS = {
    1: [255, 0, 255],   # Tím (Main)
    2: [0, 255, 255],   # Vàng (Other)
    3: [0, 165, 255]    # Cam (Turn)
}

# Cấu hình Waypoint (Giữ nguyên)
WAYPOINT_COLOR = (255, 255, 255) 
WAYPOINT_INTERVAL = 10           
ALPHA = 0.5         # Độ trong suốt overlay
MIN_AREA = 100      # Lọc vùng nhiễu nhỏ
SAMPLE_STEP = 10    # Bước nhảy khi lấy mẫu điểm

# --- 2. CLASS DỰ ĐOÁN (L OAD .PT MODEL) ---
class LanePredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔄 Đang tải model TorchScript: {model_path}")
        
        try:
            # Load model .pt (đã bao gồm cả kiến trúc và weight)
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            
            # Tối ưu hóa cho phần cứng
            if self.device.type == 'cuda':
                self.model = torch.jit.optimize_for_inference(self.model)
                torch.backends.cudnn.benchmark = True
            
            # Warm-up (chạy thử 1 lần để khởi động GPU)
            print("🔥 Warming up GPU...")
            dummy = torch.zeros((1, MODEL_H, MODEL_W, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad(): 
                self.model(dummy)
            print("✅ Model sẵn sàng!")
            
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            print("   Hãy đảm bảo file .pt tồn tại và được export đúng cách.")
            exit()

    def predict(self, img_bgr):
        # Resize về kích thước model yêu cầu
        img_resized = cv2.resize(img_bgr, (MODEL_W, MODEL_H))
        
        # Chuyển sang Tensor (H, W, C) -> (1, H, W, C)
        # Lưu ý: Model Wrapper (.pt) sẽ tự xử lý permute và normalize bên trong
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.model(img_tensor) # Output: (1, Num_Classes, H, W)
            
        # Lấy class có xác suất cao nhất -> (H, W)
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return pred_mask

# --- 3. XỬ LÝ PATH (GIỮ NGUYÊN LOGIC CŨ) ---
def get_poly_points(sample_y, sample_x, scale_x, scale_y, degree=3):
    if len(sample_y) < 4: return None
    try:
        fit = np.polyfit(sample_y, sample_x, degree)
        poly = np.poly1d(fit)
        y_min, y_max = min(sample_y), max(sample_y)
        plot_y = np.linspace(y_min, y_max, num=int(y_max - y_min))
        plot_x = poly(plot_y)
        
        # Scale về kích thước gốc của video
        plot_x_scaled = plot_x * scale_x
        plot_y_scaled = plot_y * scale_y
        
        pts_float = np.transpose(np.vstack([plot_x_scaled, plot_y_scaled]))
        return pts_float
    except:
        return None

def process_main_lane(mask_small, scale_x, scale_y):
    # Lấy class 1 (Main Lane)
    binary_mask = (mask_small == 1).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)) 
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
    # Chỉ lấy vùng lớn nhất
    largest_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) < MIN_AREA * 2: return []
    
    clean_mask = np.zeros_like(binary_mask)
    cv2.drawContours(clean_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
    
    sample_x, sample_y = [], []
    x, y, w, h = cv2.boundingRect(largest_cnt)
    
    for r in range(y + h - 1, y, -SAMPLE_STEP):
        row = clean_mask[r, x : x + w]
        indices = np.where(row == 255)[0]
        if len(indices) > 0:
            center_x = x + int(np.mean(indices))
            sample_x.append(center_x)
            sample_y.append(r)
            
    pts = get_poly_points(sample_y, sample_x, scale_x, scale_y, degree=3)
    return [pts] if pts is not None else []

def process_other_lanes(mask_small, cls_id, scale_x, scale_y):
    binary_mask = (mask_small == cls_id).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    paths = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA: continue
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        sample_x, sample_y = [], []
        
        for r in range(y + h - 1, y, -SAMPLE_STEP):
            row_slice = labels[r, x : x + w]
            indices = np.where(row_slice == i)[0]
            if len(indices) > 0:
                sample_x.append(x + int(np.mean(indices)))
                sample_y.append(r)
        
        pts = get_poly_points(sample_y, sample_x, scale_x, scale_y, degree=3)
        if pts is not None: paths.append(pts)
            
    return paths

# --- 4. VẼ VÀ HIỂN THỊ (GIỮ NGUYÊN WAYPOINT) ---
def draw_final_result(frame, mask_small, all_paths, fps):
    h_orig, w_orig = frame.shape[:2]
    
    # 1. Vẽ Overlay (Segmentation Mask)
    mask_large = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros_like(frame)
    unique_ids = np.unique(mask_small)
    
    for cls_id in unique_ids:
        if cls_id in OVERLAY_COLORS:
            color_mask[mask_large == cls_id] = OVERLAY_COLORS[cls_id]
            
    mask_bool = (mask_large > 0)
    if np.any(mask_bool):
        frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1-ALPHA, color_mask[mask_bool], ALPHA, 0)
        
    # 2. Vẽ Path & Waypoints
    for cls_id, paths_list in all_paths.items():
        color = PATH_COLORS.get(cls_id, [255, 255, 255])
        
        for pts_float in paths_list:
            pts_int = np.array([pts_float], np.int32)
            
            # Vẽ đường Path
            if cls_id == 1: 
                cv2.polylines(frame, pts_int, isClosed=False, color=(0,0,0), thickness=8) # Viền đen
                cv2.polylines(frame, pts_int, isClosed=False, color=color, thickness=4)
            else: 
                cv2.polylines(frame, pts_int, isClosed=False, color=color, thickness=3)
            
            # --- VẼ WAYPOINTS (GIỮ NGUYÊN) ---
            # pts_float: (N, 2) -> Đảo ngược để vẽ từ gần ra xa
            pts_reversed = pts_float[::-1]
            
            for i, pt in enumerate(pts_reversed):
                # Chỉ vẽ mỗi điểm thứ N
                if i % WAYPOINT_INTERVAL == 0:
                    center = (int(pt[0]), int(pt[1]))
                    # Chấm tròn trắng + viền đen
                    cv2.circle(frame, center, 4, WAYPOINT_COLOR, -1) 
                    cv2.circle(frame, center, 5, (0,0,0), 1)

    # 3. Hiển thị FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# --- 5. CHƯƠNG TRÌNH CHÍNH ---
def run():
    print("🚀 Bắt đầu nhận diện làn đường...")
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Không tìm thấy file video: {INPUT_VIDEO}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(INPUT_VIDEO)}")
    
    # Khởi tạo predictor
    predictor = LanePredictor(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tỉ lệ scale từ model ra màn hình
    scale_x = width / MODEL_W
    scale_y = height / MODEL_H
    
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))
    
    prev_time = time.time()
    pbar = tqdm(total=total_frames, desc="Processing")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Dự đoán
            mask_small = predictor.predict(frame)
            
            # 2. Xử lý hậu kỳ (tìm path)
            all_paths = {}
            unique_classes = np.unique(mask_small)
            
            if 1 in unique_classes:
                all_paths[1] = process_main_lane(mask_small, scale_x, scale_y)
            
            for cls_id in [2, 3]:
                if cls_id in unique_classes:
                    all_paths[cls_id] = process_other_lanes(mask_small, cls_id, scale_x, scale_y)
            
            # 3. Vẽ kết quả
            curr_time = time.time()
            fps_proc = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            result_frame = draw_final_result(frame, mask_small, all_paths, fps_proc)
            
            # 4. Lưu và Hiển thị
            out.write(result_frame)
            
            # Resize cửa sổ hiển thị cho dễ nhìn
            display_frame = cv2.resize(result_frame, (1024, 576))
            cv2.imshow("DSUnet Lane Detection (.pt)", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            pbar.update(1)
            
    except KeyboardInterrupt:
        print("\n⛔ Dừng bởi người dùng.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Hoàn tất! Video đã lưu tại:\n   {save_path}")

if __name__ == "__main__":
    run()