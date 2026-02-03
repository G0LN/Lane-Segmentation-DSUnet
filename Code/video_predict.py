import torch
import cv2
import numpy as np
import os
import time
import argparse

# --- 1. USER CONFIGURATION ---
MODEL_PATH = r"E:\DSUnet-drivable_area_segmentation\Experiments\DSUnet_Deploy_Fixed_20251216_210448\models\dsunet_deploy_f1_0.879.pt"
INPUT_VIDEO_PATH = r"E:\DSUnet-drivable_area_segmentation\Data\Video\Video_1(24fps).mp4"
OUTPUT_ROOT = "inference_results_video"
THRESHOLD = 0.5189

# OPTIMIZATION FLAGS
USE_FP16 = True        # Use Half Precision (Faster on modern GPUs)
SHOW_WINDOW = True    # Disable cv2.imshow to run faster
TARGET_W = 640         # Process visualization at this width (instead of 1920)
TARGET_H = 360         # Process visualization at this height

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
            dummy_dtype = torch.float16 if self.use_fp16 else torch.float32 # Raw model expects float inputs usually if internal casting isn't robust
            # Note: The deployable model expects input to be resized/normalized internally usually from uint8/float32.
            # However, for pure speed, we assume the TorchScript model handles resizing.
            # But passing uint8 to .half() model might cause issues, so we stick to standard flow
            # For deployable wrapper which takes uint8, we keep input as uint8, but internal weights are FP16
            
            # Simple warmup
            dummy = torch.zeros((1, 360, 640, 3), dtype=torch.uint8).to(self.device)
            with torch.no_grad():
                # Autocast handles the precision mismatch if model expects float but we pass uint8 wrapper
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    self.model(dummy)
            print("✅ Model is ready!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()

    def predict(self, img_bgr):
        # 1. Resize input explicitly to reduce CPU->GPU bandwidth
        # Resizing to model target size (512x288) externally is faster than sending 1080p
        # But to keep visualization good, we resize to TARGET_W/H defined in config
        
        # We pass the image directly. The Tensor conversion is fast.
        img_tensor = torch.from_numpy(img_bgr).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                prob_map = self.model(img_tensor)
            
        # 3. Post-processing
        prob_map = prob_map.squeeze().cpu().numpy()
        
        # We do NOT resize back to 1080p here to save time.
        # We will resize the visualization frame instead.
        return prob_map

# --- 3. UTILS ---
def process_video(detector, video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"❌ Input video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will output video at TARGET resolution, NOT original 1080p
    # This is a common trade-off for speed.
    print(f"🎬 Processing Video: {os.path.basename(video_path)}")
    print(f"   - Target Resolution: {TARGET_W}x{TARGET_H}")

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_fast.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps_input, (TARGET_W, TARGET_H))

    frame_count = 0
    start_time = time.time()
    prev_frame_time = 0
    
    print("▶ Starting optimized inference...")
    
    # Pre-allocate memory for green layer to avoid creating it every frame
    green_layer = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    green_layer[:, :, 1] = 255 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        new_frame_time = time.time()
        
        # --- OPTIMIZATION: Resize Input Frame FIRST ---
        # Instead of processing 1080p, we resize the raw frame to 640x360
        # This speeds up everything (tensor conversion, transfer, visualization)
        frame_resized = cv2.resize(frame, (TARGET_W, TARGET_H))
        
        # Predict
        # The model returns 288x512 mask
        prob_map = detector.predict(frame_resized)
        
        # Resize mask to match our visualization frame (TARGET_W x TARGET_H)
        prob_map_resized = cv2.resize(prob_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        
        # Threshold
        mask_bool = prob_map_resized > detector.threshold
        
        # Fast Overlay using Numpy indexing (Avoid cv2.addWeighted for full image)
        # Only modify pixels that are lane
        # alpha = 0.4 -> 0.6 * original + 0.4 * green
        
        # Convert to more efficient blending
        # frame_resized[mask_bool] = frame_resized[mask_bool] * 0.6 + green_layer[mask_bool] * 0.4
        # A simpler way without float math for speed:
        # Just add green tint manually
        frame_resized[mask_bool, 1] = 255 # Set Green channel to max
        # Or if you want transparency, stick to cv2.addWeighted but only on ROI? 
        # cv2.addWeighted is actually quite optimized in C++. Let's use it but on small image.
        
        result_frame = cv2.addWeighted(frame_resized, 1, green_layer, 0.4, 0)
        # Restore non-mask regions (optional, addWeighted blends whole image)
        # Actually addWeighted blends everything making whole image green tinted.
        # Correct way for speed:
        frame_final = frame_resized.copy()
        frame_final[mask_bool] = cv2.addWeighted(frame_resized[mask_bool], 0.6, green_layer[mask_bool], 0.4, 0)

        # FPS
        fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
        prev_frame_time = new_frame_time
        
        cv2.putText(frame_final, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame_final)
        
        if SHOW_WINDOW:
            cv2.imshow('Lane Detection Fast', frame_final)
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