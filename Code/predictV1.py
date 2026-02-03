import torch
import cv2
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CẤU HÌNH (CONFIGURATION) ---
# Thay đổi đường dẫn model và ảnh của bạn tại đây
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_Deploy_Fixed_20251216_210448\models\dsunet_deploy_f1_0.879.pt"
INPUT_PATH = r"E:\DSUnet-drivable_area_segmentation\Data\test\c1226eb9-27ecd843.jpg"
OUTPUT_ROOT = "inference_cubic_paths"
THRESHOLD = 0.5189
MIN_LANE_AREA = 2000  # Bỏ qua các vùng nhiễu nhỏ hơn 500 pixel

# --- 2. LANE DETECTOR (KHÔNG ĐỔI) ---
class LaneDetector:
    def __init__(self, model_path, threshold=0.5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            dummy = torch.zeros((1, 360, 640, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad(): self.model(dummy)
            print("✅ Đã tải model thành công!")
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")

    def predict(self, img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        img_tensor = torch.from_numpy(img_bgr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob_map = self.model(img_tensor)
        prob_map = prob_map.squeeze().cpu().numpy()
        prob_map_resized = cv2.resize(prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        # Tạo mask nhị phân: 255 là làn đường, 0 là nền
        mask = (prob_map_resized > self.threshold).astype(np.uint8) * 255
        return mask

# --- 3. LOGIC TÍNH TOÁN PATH BẬC 3 (QUAN TRỌNG) ---
def calculate_cubic_path(instance_mask, sample_step=10):
    """
    Tính toán đường Path Bậc 3 cho MỘT vùng làn đường cụ thể.
    """
    h, w = instance_mask.shape
    sample_x = []
    sample_y = []
    
    # Quét từ dưới lên trên (Bottom-up), nhảy cóc mỗi 10 dòng
    for y in range(h - 1, 0, -sample_step):
        row = instance_mask[y, :]
        # Tìm các pixel thuộc vùng mask này
        indices = np.where(row == 255)[0]
        
        if len(indices) > 0:
            # Logic "Segmentation Mask":
            # Tâm của làn đường là trung điểm của đoạn pixel trắng
            x_start = indices[0]   # Mép trái
            x_end = indices[-1]    # Mép phải
            center_x = int((x_start + x_end) / 2)
            
            sample_x.append(center_x)
            sample_y.append(y)
            
    # Đa thức bậc 3 cần tối thiểu 4 điểm để giải phương trình
    if len(sample_y) < 4:
        return None

    try:
        # Fit đa thức bậc 3: x = ay^3 + by^2 + cy + d
        # Biến độc lập là Y, biến phụ thuộc là X (vì đường cong dọc)
        fit_params = np.polyfit(sample_y, sample_x, 3)
        poly_func = np.poly1d(fit_params)
        
        # Tạo tập điểm Y dày đặc để vẽ đường cong mượt mà
        y_min = min(sample_y)
        y_max = max(sample_y)
        plot_y = np.linspace(y_min, y_max, num=int(y_max - y_min))
        
        # Tính X tương ứng
        plot_x = poly_func(plot_y)
        
        # Gom lại thành mảng điểm để vẽ (OpenCV format)
        curve_pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], np.int32)
        return curve_pts
    except Exception as e:
        # Trường hợp fit lỗi (VD: đường thẳng đứng hoàn toàn)
        return None

# --- 4. XỬ LÝ CHÍNH & HIỂN THỊ ---
def process_lanes_and_draw(img_bgr, mask_binary):
    """
    Tách các làn đường -> Tính Path riêng cho từng làn -> Vẽ kết quả.
    """
    # Bước 1: Phân tách các vùng làn đường rời rạc (Instance Segmentation giả lập)
    # num_labels: Tổng số vùng
    # labels_im: Ảnh mask với ID từng vùng (0=nền, 1=làn A, 2=làn B...)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    vis_img = img_bgr.copy()
    
    # Tạo màu ngẫu nhiên để tô các làn đường (để dễ phân biệt)
    colors = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)
    
    # Duyệt qua từng vùng (Bỏ qua label 0 là background)
    print(f"--> Tìm thấy {num_labels - 1} vùng ứng viên.")
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Lọc nhiễu: Bỏ qua các vùng quá nhỏ
        if area < MIN_LANE_AREA:
            continue
            
        # 1. Tạo mask riêng cho làn đường thứ i
        instance_mask = np.zeros_like(mask_binary)
        instance_mask[labels_im == i] = 255
        
        # 2. Tính Path Bậc 3 cho làn đường này
        path_points = calculate_cubic_path(instance_mask, sample_step=10)
        
        # 3. Vẽ kết quả
        # A. Tô màu vùng mask (Overlay mờ)
        color = colors[i].tolist()
        colored_mask = np.zeros_like(vis_img)
        colored_mask[labels_im == i] = color
        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.4, 0) # Alpha = 0.4
        
        # B. Vẽ đường Path Prediction (Màu Đỏ Đậm)
        if path_points is not None:
            cv2.polylines(vis_img, path_points, isClosed=False, color=(0, 0, 255), thickness=4)
            
            # (Optional) Vẽ mũi tên chỉ hướng ở đầu đường path
            if len(path_points[0]) > 20:
                 end_pt = tuple(path_points[0][-1])      # Điểm gần xe nhất
                 near_pt = tuple(path_points[0][-15])    # Điểm xa hơn chút
                 cv2.arrowedLine(vis_img, end_pt, near_pt, (0, 255, 255), 3)

    return vis_img

def main():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Không tìm thấy file model: {MODEL_PATH}")
        # Vẫn chạy tiếp nếu bạn muốn test code logic (cần tự sửa logic load ảnh)
    
    try:
        detector = LaneDetector(MODEL_PATH, threshold=THRESHOLD)
    except:
        print("Không thể khởi tạo Detector.")
        return

    # 2. Xử lý ảnh đầu vào
    target_files = []
    if os.path.isfile(INPUT_PATH): 
        target_files.append(INPUT_PATH)
    elif os.path.isdir(INPUT_PATH): 
        target_files = glob.glob(os.path.join(INPUT_PATH, "*.jpg"))
    
    if not target_files:
        print("Không tìm thấy ảnh đầu vào!")
        return

    print(f"🚀 Bắt đầu xử lý {len(target_files)} ảnh...")
    
    # Tạo thư mục lưu
    save_dir = os.path.join(OUTPUT_ROOT, "results")
    os.makedirs(save_dir, exist_ok=True)

    for fpath in tqdm(target_files):
        img = cv2.imread(fpath)
        if img is None: continue
        
        # Predict
        mask = detector.predict(img)
        
        # Process & Draw Cubic Paths
        result = process_lanes_and_draw(img, mask)
        
        # Save
        fname = os.path.basename(fpath)
        cv2.imwrite(os.path.join(save_dir, f"cubic_{fname}"), result)
        
    print(f"✅ Hoàn tất! Kết quả tại: {save_dir}")

# --- 5. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    main()