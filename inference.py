import torch
import cv2
import numpy as np
import yaml
import time
from PIL import Image
from models import DSUnet
from utils import load_checkpoint

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

def predict_image(image_path, model, device, img_height, img_width, lane_threshold=0.15):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((img_width, img_height))
    
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        
    # Process output using Softmax for smart lane-priority override
    probs = torch.softmax(output.squeeze(0), dim=0).cpu().numpy() # (num_classes, H, W)
    pred = np.argmax(probs, axis=0) # Default argmax
    
    if lane_threshold is not None and lane_threshold > 0:
        # Lane classes are 1 to 7 (inclusive)
        lane_probs = probs[1:8, :, :]
        max_lane_prob = np.max(lane_probs, axis=0)
        best_lane_class = np.argmax(lane_probs, axis=0) + 1
        
        # Override vehicle (8) or background (0) if a lane class has sufficient probability
        override_condition = ((pred == 8) | (pred == 0)) & (max_lane_prob > lane_threshold)
        pred[override_condition] = best_lane_class[override_condition]
    
    # Map predictions to colors
    pred_colored = COLORS[pred]
    
    # Resize back to original
    pred_colored = cv2.resize(pred_colored, original_size, interpolation=cv2.INTER_NEAREST)
    
    return image_np, pred_colored

def inference(config_path, checkpoint_path, image_path, output_path, lane_threshold=0.15):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=config['model']['num_classes'],
        dropout=config['model'].get('dropout', 0.5)
    ).to(device)
    
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    print(f"Running inference on {image_path} with lane_threshold={lane_threshold}...")
    original_img, mask_img = predict_image(
        image_path, model, device, 
        config['dataset']['image_height'], 
        config['dataset']['image_width'],
        lane_threshold=lane_threshold
    )
    
    # Blend image and mask
    blended = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.5, mask_img, 0.5, 0)
    cv2.imwrite(output_path, blended)
    print(f"Saved result to {output_path}")

def inference_video(config_path, checkpoint_path, video_path, output_path, lane_threshold=0.15):
    from tqdm import tqdm
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    img_height = config['dataset']['image_height']
    img_width = config['dataset']['image_width']
    
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=config['model']['num_classes'],
        dropout=config['model'].get('dropout', 0.5)
    ).to(device)
    
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing Video: {video_path}")
    print(f"Resolution: {width}x{height} | Original FPS: {fps:.1f} | Total Frames: {total_frames}")
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    pbar = tqdm(total=total_frames, desc="Inference Video")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Preprocess BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_width, img_height))
        
        frame_tensor = torch.from_numpy(frame_resized.transpose((2, 0, 1))).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(frame_tensor)
            
        # Smart Lane priority thresholding
        probs = torch.softmax(output.squeeze(0), dim=0).cpu().numpy()
        pred = np.argmax(probs, axis=0)
        
        if lane_threshold is not None and lane_threshold > 0:
            lane_probs = probs[1:8, :, :]
            max_lane_prob = np.max(lane_probs, axis=0)
            best_lane_class = np.argmax(lane_probs, axis=0) + 1
            
            override_condition = ((pred == 8) | (pred == 0)) & (max_lane_prob > lane_threshold)
            pred[override_condition] = best_lane_class[override_condition]
            
        pred_colored = COLORS[pred]
        pred_colored = cv2.resize(pred_colored, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Blend frame and mask
        blended = cv2.addWeighted(frame, 0.5, pred_colored, 0.5, 0)
        
        # Calculate actual FPS
        elapsed_time = time.time() - start_time
        curr_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
        
        # Write FPS on frame
        fps_text = f"FPS: {curr_fps:.1f}"
        cv2.putText(blended, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        
        out.write(blended)
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()
    print(f"✅ Video processing completed successfully! Saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DSUnet Inference")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth')
    parser.add_argument('--input', type=str, required=True, help="Path to input image or video")
    parser.add_argument('--output', type=str, required=True, help="Path to save result")
    parser.add_argument('--video', action='store_true', help="Set this flag if input is a video")
    parser.add_argument('--lane_threshold', type=float, default=0.15, help="Threshold to prioritize lanes over vehicles/background")
    
    args = parser.parse_args()
    
    # Auto-detect if input is a video based on file extension
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.webm')
    is_video = args.video or args.input.lower().endswith(video_extensions)
    
    if is_video:
        inference_video(args.config, args.checkpoint, args.input, args.output, args.lane_threshold)
    else:
        inference(args.config, args.checkpoint, args.input, args.output, args.lane_threshold)
