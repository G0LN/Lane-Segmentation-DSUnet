import torch
import cv2
import numpy as np
import yaml
import time
import os
import re
from PIL import Image

from models import DSUnet
from utils import load_checkpoint

class InferenceEngine:
    """
    Object-Oriented InferenceEngine class for B-DSUnet.
    Encapsulates all inference logic, color mapping, model initialization,
    lane priority override strategies, and single-image or video rendering.
    """
    # Color map for visualizing classes (0: background, 1: continuous white, 2: continuous yellow, 3: dashed, 4: double continuous yellow, 5: main-lane, 6: other-lane, 7: turn-lane, 8: vehicle)
    COLORS = np.array([
        [0, 0, 0],         # 0: Background/new-tusimple - Black
        [255, 255, 255],   # 1: continuous white - White
        [255, 255, 0],     # 2: continuous yellow - Yellow
        [128, 128, 128],   # 3: dashed - Gray
        [255, 165, 0],     # 4: double continuous yellow - Orange
        [0, 255, 0],       # 5: main-lane - Green
        [0, 0, 255],       # 6: other-lane - Blue
        [255, 0, 255],     # 7: turn-lane - Magenta
        [0, 255, 255]      # 8: vehicle - Cyan
    ], dtype=np.uint8)

    def __init__(self, config_path="configs/default.yaml", checkpoint_path="checkpoints/model_best.pth", lane_threshold=0.15):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.lane_threshold = lane_threshold
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['training']['device'] if torch.cuda.is_available() else "cpu")
        self.img_height = self.config['dataset']['image_height']
        self.img_width = self.config['dataset']['image_width']
        self.num_classes = self.config['model']['num_classes']
        
        self._resolve_checkpoint_path()
        self._setup_model()

    def _resolve_checkpoint_path(self):
        # Auto-detect latest checkpoint if the specified path doesn't exist
        if not os.path.exists(self.checkpoint_path):
            base_dir = os.path.dirname(self.checkpoint_path)
            filename = os.path.basename(self.checkpoint_path)
            if not base_dir or base_dir == '':
                base_dir = "checkpoints"
                
            max_num = 0
            latest_dir = None
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    if os.path.isdir(os.path.join(base_dir, item)):
                        match = re.match(r"^checkpoint(\d+)$", item)
                        if match:
                            num = int(match.group(1))
                            # Only select if the target file actually exists in this folder
                            if os.path.exists(os.path.join(base_dir, item, filename)):
                                if num > max_num:
                                    max_num = num
                                    latest_dir = os.path.join(base_dir, item)
            if latest_dir is not None:
                possible_path = os.path.join(latest_dir, filename)
                print(f"Checkpoint not found at '{self.checkpoint_path}'. Automatically using the latest found at: '{possible_path}'")
                self.checkpoint_path = possible_path

    def _setup_model(self):
        width_multiplier = self.config['model'].get('width_multiplier', self.config['model'].get('alpha', 1.0))
        print(f"Model initialization: Width Multiplier (Alpha) = {width_multiplier}")
        
        self.model = DSUnet(
            in_channels=self.config['model']['in_channels'], 
            num_classes=self.num_classes,
            dropout=self.config['model'].get('dropout', 0.5),
            width_multiplier=width_multiplier
        ).to(self.device)
        
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        load_checkpoint(self.checkpoint_path, self.model)
        self.model.eval()
        self.model.switch_to_deploy() # Benchmark actual deploy/fused state

    def predict_image(self, original_img):
        """
        Runs forward pass on a single NumPy HWC BGR/RGB image array.
        Handles resizing, normalizations, batch dimension, forward pass, Softmax,
        lane priority overrides, color mapping, and resizing back to original size.
        """
        original_size = (original_img.shape[1], original_img.shape[0]) # (width, height)
        
        # Preprocess image using exact same resizing as training (OpenCV INTER_LINEAR)
        image_resized = cv2.resize(original_img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device) # Add batch dimension
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            
        # Process output using Softmax for smart lane-priority override
        probs = torch.softmax(output.squeeze(0), dim=0).cpu().numpy() # (num_classes, H, W)
        pred = np.argmax(probs, axis=0) # Default argmax
        
        if self.lane_threshold is not None and self.lane_threshold > 0:
            # Lane classes are 1 to 7 (inclusive)
            lane_probs = probs[1:8, :, :]
            max_lane_prob = np.max(lane_probs, axis=0)
            best_lane_class = np.argmax(lane_probs, axis=0) + 1
            
            # Override vehicle (8) or background (0) if a lane class has sufficient probability
            override_condition = ((pred == 8) | (pred == 0)) & (max_lane_prob > self.lane_threshold)
            pred[override_condition] = best_lane_class[override_condition]
        
        # Map predictions to colors
        pred_colored = self.COLORS[pred]
        
        # Resize back to original size using NEAREST interpolation to keep labels crisp
        pred_colored = cv2.resize(pred_colored, original_size, interpolation=cv2.INTER_NEAREST)
        
        return pred_colored

    def infer_single_image(self, image_path, output_path):
        print(f"\n[Running] Image Inference: {image_path}...")
        
        # Load original image as RGB
        original_img_pil = Image.open(image_path).convert("RGB")
        original_img = np.array(original_img_pil)
        
        pred_colored = self.predict_image(original_img)
        
        # Convert mask_img from RGB to BGR before blending with BGR image
        mask_bgr = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)
        
        # Blend image and mask at original resolution
        blended = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.5, mask_bgr, 0.5, 0)
        cv2.imwrite(output_path, blended)
        print(f"[Success] Saved blended result to: {output_path}")

    def infer_video(self, video_path, output_path):
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[Running] Video Inference: {video_path}")
        print(f"  - Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")
        
        # Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        pbar = tqdm(total=total_frames, desc="Inference Video")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Preprocess BGR frame to RGB for model ingestion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_colored_rgb = self.predict_image(frame_rgb)
            
            # Convert predicted color map to BGR for video writing
            pred_colored_bgr = cv2.cvtColor(pred_colored_rgb, cv2.COLOR_RGB2BGR)
            
            # Blend frame and mask
            blended = cv2.addWeighted(frame, 0.5, pred_colored_bgr, 0.5, 0)
            
            # Calculate actual processing FPS
            elapsed_time = time.time() - start_time
            curr_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            
            # Burn FPS overlay onto output frame
            fps_text = f"FPS: {curr_fps:.1f}"
            cv2.putText(blended, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
            
            out.write(blended)
            pbar.update(1)
            
        cap.release()
        out.release()
        pbar.close()
        print(f"[Success] Video inference completed successfully! Saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DSUnet Inference Engine")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth')
    parser.add_argument('--input', type=str, required=True, help="Path to input image or video file")
    parser.add_argument('--output', type=str, required=True, help="Path to save output result file")
    parser.add_argument('--video', action='store_true', help="Force treating input file as video")
    parser.add_argument('--lane_threshold', type=float, default=0.15, help="Softmax threshold to prioritize lane over vehicle/bg")
    
    args = parser.parse_args()
    
    engine = InferenceEngine(
        config_path=args.config, 
        checkpoint_path=args.checkpoint, 
        lane_threshold=args.lane_threshold
    )
    
    # Auto-detect if input is a video based on file extension
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.webm')
    is_video = args.video or args.input.lower().endswith(video_extensions)
    
    if is_video:
        engine.infer_video(args.input, args.output)
    else:
        engine.infer_single_image(args.input, args.output)
