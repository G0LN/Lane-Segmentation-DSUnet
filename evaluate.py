import yaml
import torch
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
    
    # Initialize and load model
    model = DSUnet(
        in_channels=config['model']['in_channels'], 
        num_classes=num_classes
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
    # Example usage
    # evaluate("configs/default.yaml", "checkpoints/model_best.pth")
    pass
