import torch
import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# --- 1. CẤU HÌNH (CONFIGURATION) ---
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_test20260122_003558\models\dsunet_deploy.pt"
INPUT_VIDEO = r"E:\DSUnet-drivable_area_segmentation\Data\Video\Video_1(24fps).mp4"
OUTPUT_DIR = r"E:\DSUnet-drivable_area_segmentation\Inference_Result_Waypoints2"

# Kích thước xử lý
MODEL_W, MODEL_H = 512, 288

# Màu sắc (BGR)
# 1: Main Lane, 2: Other, 3: Turn
OVERLAY_COLORS = {
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 0, 0]
}

PATH_COLORS = {
    1: [255, 0, 255],   # Main Path: Tím
    2: [0, 255, 255],   # Other Path: Vàng
    3: [0, 165, 255]    # Turn Path: Cam
}

# Màu Waypoint (Chấm tròn trên đường path)
WAYPOINT_COLOR = (255, 255, 255) # Màu trắng
WAYPOINT_INTERVAL = 10           # Vẽ waypoint mỗi 10 pixel (trên trục Y)

ALPHA = 0.5         # Độ trong suốt
MIN_AREA = 100      # Lọc nhiễu
SAMPLE_STEP = 10    # Bước nhảy quét

# --- 2. CLASS DỰ ĐOÁN ---
class LanePredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔄 Đang tải model: {model_path}")
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            dummy = torch.zeros((1, MODEL_H, MODEL_W, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad(): self.model(dummy)
            print("✅ Model sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            exit()

    def predict(self, img_bgr):
        img_resized = cv2.resize(img_bgr, (MODEL_W, MODEL_H))
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model(img_tensor)
        pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return pred_mask

# --- 3. XỬ LÝ PATH ---

def get_poly_points(sample_y, sample_x, scale_x, scale_y, degree=3):
    if len(sample_y) < 4: return None
    try:
        fit = np.polyfit(sample_y, sample_x, degree)
        poly = np.poly1d(fit)
        
        y_min, y_max = min(sample_y), max(sample_y)
        
        # Tạo mảng Y dày để vẽ
        plot_y = np.linspace(y_min, y_max, num=int(y_max - y_min)) # Num = độ dài pixel Y
        plot_x = poly(plot_y)
        
        # Scale
        plot_x_scaled = plot_x * scale_x
        plot_y_scaled = plot_y * scale_y
        
        # Trả về mảng điểm (N, 2) float để có thể xử lý waypoint sau này
        pts_float = np.transpose(np.vstack([plot_x_scaled, plot_y_scaled]))
        return pts_float
    except:
        return None

def process_main_lane(mask_small, scale_x, scale_y):
    binary_mask = (mask_small == 1).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)) 
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
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

# --- 4. VẼ VÀ HIỂN THỊ (CÓ WAYPOINT) ---
def draw_final_result(frame, mask_small, all_paths, fps):
    h_orig, w_orig = frame.shape[:2]
    
    # 1. Overlay
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
            # Chuyển sang int để vẽ line
            pts_int = np.array([pts_float], np.int32)
            
            if cls_id == 1: # Main Lane
                cv2.polylines(frame, pts_int, isClosed=False, color=(0,0,0), thickness=8)
                cv2.polylines(frame, pts_int, isClosed=False, color=color, thickness=4)
            else: # Other Lanes
                cv2.polylines(frame, pts_int, isClosed=False, color=color, thickness=3)
            
            # --- VẼ WAYPOINTS ---
            # Duyệt qua các điểm trên path, lấy mẫu mỗi N pixel (WAYPOINT_INTERVAL)
            # pts_float shape là (N, 2) -> (x, y)
            # Chúng ta vẽ ngược từ dưới lên trên (gần xe ra xa)
            
            # Đảo ngược mảng để điểm đầu tiên là điểm gần xe nhất (y lớn nhất)
            pts_reversed = pts_float[::-1]
            
            for i, pt in enumerate(pts_reversed):
                # Vẽ waypoint mỗi 10 điểm (tương ứng khoảng cách dọc trục Y do linspace tạo ra)
                if i % WAYPOINT_INTERVAL == 0:
                    center = (int(pt[0]), int(pt[1]))
                    
                    # Vẽ chấm tròn
                    cv2.circle(frame, center, 4, WAYPOINT_COLOR, -1) # Chấm trắng
                    # Vẽ viền cho chấm tròn để nổi bật
                    cv2.circle(frame, center, 5, (0,0,0), 1)

    # 3. FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# --- 5. MAIN ---
def run():
    if not os.path.exists(INPUT_VIDEO):
        print("❌ Không tìm thấy video.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"waypoints_{os.path.basename(INPUT_VIDEO)}")
    
    predictor = LanePredictor(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    
    scale_x = width / MODEL_W
    scale_y = height / MODEL_H
    
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))
    
    prev_time = time.time()
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        mask_small = predictor.predict(frame)
        all_paths = {}
        
        if 1 in np.unique(mask_small):
            all_paths[1] = process_main_lane(mask_small, scale_x, scale_y)
            
        for cls_id in [2, 3]:
            if cls_id in np.unique(mask_small):
                all_paths[cls_id] = process_other_lanes(mask_small, cls_id, scale_x, scale_y)
        
        curr_time = time.time()
        fps_proc = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        result_frame = draw_final_result(frame, mask_small, all_paths, fps_proc)
        out.write(result_frame)
        
        cv2.imshow("Lane Path with Waypoints", cv2.resize(result_frame, (1024, 576)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        pbar.update(1)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Video đã lưu tại: {save_path}")

if __name__ == "__main__":
    run()