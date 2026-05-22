import yaml
import torch
import os
import re
from tqdm import tqdm

from models import DSUnet
from data.dataset import LaneSegmentationDataset
from data.transforms import get_val_transforms
from torch.utils.data import DataLoader
from utils import get_metrics, load_checkpoint

def evaluate(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    num_classes = config['model']['num_classes']
    
    # Auto-detect latest checkpoint if the specified path doesn't exist
    if not os.path.exists(checkpoint_path):
        base_dir = os.path.dirname(checkpoint_path)
        filename = os.path.basename(checkpoint_path)
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
            print(f"Checkpoint not found at '{checkpoint_path}'. Automatically using the latest checkpoint found at: '{possible_path}'")
            checkpoint_path = possible_path
    
    # Load dataset
    val_transform = get_val_transforms(
        config['dataset']['image_height'], 
        config['dataset']['image_width']
    )
    test_dataset = LaneSegmentationDataset(
        images_dir=config['dataset']['test_images_dir'],
        json_path=config['dataset']['test_json_path'],
        img_height=config['dataset']['image_height'],
        img_width=config['dataset']['image_width'],
        transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Width Scaling / Alpha
    width_multiplier = config['model'].get('width_multiplier', config['model'].get('alpha', 1.0))
    print(f"Model initialization: Width Multiplier (Alpha) = {width_multiplier}")
    
    # Initialize and load model
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=num_classes,
        dropout=config['model'].get('dropout', 0.5),
        width_multiplier=width_multiplier
    ).to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    all_preds = []
    all_masks = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu())
            all_masks.append(masks)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    metrics = get_metrics(all_preds, all_masks, num_classes)
    print(f"Evaluation Results - mIoU: {metrics['mIoU']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DSUnet Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth', help='Path to checkpoint file')
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)
