import torch
import cv2
import numpy as np
import yaml
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

def predict_image(image_path, model, device, img_height, img_width):
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
        
    # Process output
    pred = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
    
    # Map predictions to colors
    pred_colored = COLORS[pred]
    
    # Resize back to original
    pred_colored = cv2.resize(pred_colored, original_size, interpolation=cv2.INTER_NEAREST)
    
    return image_np, pred_colored

def inference(config_path, checkpoint_path, image_path, output_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=config['model']['num_classes']
    ).to(device)
    
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    print(f"Running inference on {image_path}...")
    original_img, mask_img = predict_image(
        image_path, model, device, 
        config['dataset']['image_height'], 
        config['dataset']['image_width']
    )
    
    # Blend image and mask
    blended = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.5, mask_img, 0.5, 0)
    cv2.imwrite(output_path, blended)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Example usage
    # inference("configs/default.yaml", "checkpoints/model_best.pth", "test_image.jpg", "result.jpg")
    pass
