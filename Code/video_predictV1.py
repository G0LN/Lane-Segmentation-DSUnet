import torch
import cv2
import numpy as np
import os
import time
import argparse

# --- 1. USER CONFIGURATION ---
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_Deploy_Fixed_20251216_210448\models\dsunet_deploy_f1_0.879.pt"
INPUT_VIDEO_PATH = r"E:\DSUnet-drivable_area_segmentation\Data\Video\mixkit-point-of-view-from-a-bus-passenger-seat-roading-in-4394-hd-ready.mp4"
OUTPUT_ROOT = "inference_results_video"
THRESHOLD = 0.5189

# OPTIMIZATION FLAGS
USE_FP16 = True        # Sử dụng Half Precision (Nhanh hơn trên GPU hỗ trợ)
SHOW_WINDOW = True     # Hiện cửa sổ cv2.imshow
TARGET_W = 640         # Chiều rộng xử lý (Resize input để tăng tốc)
TARGET_H = 360         # Chiều cao xử lý
MIN_LANE_AREA = 300    # Diện tích tối thiểu (pixel) để coi là 1 làn đường (trên ảnh resized)

# --- 2. LANE DETECTOR CLASS ---
class LaneDetector:
    def __init__(self, model_path, threshold=0.5, device='cuda', use_fp16=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')
        
        print(f"🔄 Loading model from: {model_path}")
        print(f"⚙  Device: {self.device} | FP16: {self.use_fp16}")
        
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            
            if self.use_fp16:
                self.model.half() # Convert model to 16-bit
            
            # Warm-up
            print("🔥 Warming up GPU...")
            dummy = torch.zeros((1, 360, 640, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    self.model(dummy)
            print("✅ Model is ready!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()

    def predict(self, img_bgr_resized):
        # Convert to Tensor [1, H, W, 3]
        img_tensor = torch.from_numpy(img_bgr_resized).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                prob_map = self.model(img_tensor)
            
        # Post-processing
        prob_map = prob_map.squeeze().cpu().numpy()
        return prob_map

# --- 3. PATH PREDICTION LOGIC (CUBIC) ---
def calculate_cubic_path(instance_mask, sample_step=10):
    """
    Tính toán đường dẫn bậc 3 cho một làn đường cụ thể.
    """
    h, w = instance_mask.shape
    sample_x = []
    sample_y = []
    
    # Quét từ dưới lên trên
    for y in range(h - 1, 0, -sample_step):
        row = instance_mask[y, :]
        indices = np.where(row == 255)[0]
        
        if len(indices) > 0:
            # Tìm tâm của làn (Center of Segment)
            x_start = indices[0]
            x_end = indices[-1]
            center_x = int((x_start + x_end) / 2)
            
            sample_x.append(center_x)
            sample_y.append(y)
            
    # Cần tối thiểu 4 điểm để fit đa thức bậc 3
    if len(sample_y) < 4:
        return None

    try:
        # Fit Cubic Polynomial: x = ay^3 + by^2 + cy + d
        fit_params = np.polyfit(sample_y, sample_x, 3)
        poly_func = np.poly1d(fit_params)
        
        # Tạo điểm vẽ
        y_min = min(sample_y)
        y_max = max(sample_y)
        plot_y = np.linspace(y_min, y_max, num=int(y_max - y_min))
        plot_x = poly_func(plot_y)
        
        curve_pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], np.int32)
        return curve_pts
    except:
        return None

# --- 4. VIDEO PROCESSING ---
def process_video(detector, video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"❌ Input video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Processing Video: {os.path.basename(video_path)}")
    print(f"   - Target Resolution: {TARGET_W}x{TARGET_H}")

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_path.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps_input, (TARGET_W, TARGET_H))

    frame_count = 0
    start_time = time.time()
    prev_frame_time = 0
    
    # Pre-allocate memory for overlays
    green_layer = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    green_layer[:, :, 1] = 255  # Green Channel

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        new_frame_time = time.time()
        
        # 1. Resize Frame (Optimization)
        frame_resized = cv2.resize(frame, (TARGET_W, TARGET_H))
        
        # 2. Predict Probability Map
        prob_map = detector.predict(frame_resized)
        
        # 3. Resize Mask to match visualization frame
        prob_map_resized = cv2.resize(prob_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        
        # 4. Create Binary Mask
        mask_binary = (prob_map_resized > detector.threshold).astype(np.uint8) * 255
        
        # --- PATH PREDICTION START ---
        # 5. Connected Components: Tách các làn đường riêng biệt
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        paths_to_draw = []
        
        # Duyệt qua từng làn đường tìm thấy
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_LANE_AREA: continue # Bỏ qua nhiễu
            
            # Tách riêng mask của làn này
            instance_mask = np.zeros_like(mask_binary)
            instance_mask[labels_im == i] = 255
            
            # Tính Cubic Path
            path = calculate_cubic_path(instance_mask, sample_step=10)
            if path is not None:
                paths_to_draw.append(path)
        # --- PATH PREDICTION END ---

        # 6. Visualization
        # A. Vẽ lớp phủ màu xanh lá (Segmentation Overlay)
        frame_final = frame_resized.copy()
        mask_bool = mask_binary > 0
        if np.any(mask_bool):
            # Blend nhanh bằng numpy indexing
            frame_final[mask_bool] = cv2.addWeighted(frame_resized[mask_bool], 0.6, green_layer[mask_bool], 0.4, 0)

        # B. Vẽ đường Path Prediction (Màu Đỏ)
        for path in paths_to_draw:
            cv2.polylines(frame_final, path, isClosed=False, color=(0, 0, 255), thickness=3)
            # Vẽ mũi tên
            if len(path[0]) > 10:
                cv2.arrowedLine(frame_final, tuple(path[0][-1]), tuple(path[0][-5]), (0, 0, 255), 2)

        # 7. FPS Counter
        fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
        prev_frame_time = new_frame_time
        cv2.putText(frame_final, f"FPS: {int(fps)} | Lanes: {len(paths_to_draw)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Write & Show
        out.write(frame_final)
        
        if SHOW_WINDOW:
            cv2.imshow('Cubic Lane Path Video', frame_final)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames... (FPS: {int(fps)})")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n✅ Complete! Saved to: {save_path}")
    print(f"⏱ Total Time: {total_time:.2f}s")
    print(f"📊 Average FPS: {frame_count / total_time:.2f}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        exit()
        
    detector = LaneDetector(MODEL_PATH, threshold=THRESHOLD, use_fp16=USE_FP16)
    process_video(detector, INPUT_VIDEO_PATH, OUTPUT_ROOT)