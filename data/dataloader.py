from torch.utils.data import DataLoader
from .dataset import LaneSegmentationDataset
from .transforms import get_train_transforms, get_val_transforms

def get_dataloaders(config):
    """
    Creates and returns train, validation dataloaders based on COCO JSON format.
    """
    img_height = config['dataset']['image_height']
    img_width = config['dataset']['image_width']
    
    # Load transforms
    train_transform = get_train_transforms(img_height, img_width)
    val_transform = get_val_transforms(img_height, img_width)

    # Initialize datasets using COCO format
    train_dataset = LaneSegmentationDataset(
        images_dir=config['dataset']['train_images_dir'],
        json_path=config['dataset']['train_json_path'],
        img_height=img_height,
        img_width=img_width,
        transform=train_transform
    )
    
    val_dataset = LaneSegmentationDataset(
        images_dir=config['dataset']['val_images_dir'],
        json_path=config['dataset']['val_json_path'],
        img_height=img_height,
        img_width=img_width,
        transform=val_transform
    )

    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader
