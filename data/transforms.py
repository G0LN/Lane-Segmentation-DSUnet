# You can use albumentations for easy image + mask augmentations
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_height, img_width):
    """
    Returns data augmentations for training.
    """
    # Example using albumentations
    # return A.Compose(
    #     [
    #         A.Resize(height=img_height, width=img_width),
    #         A.HorizontalFlip(p=0.5),
    #         A.RandomBrightnessContrast(p=0.2),
    #         A.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ]
    # )
    return None

def get_val_transforms(img_height, img_width):
    """
    Returns data augmentations for validation/testing.
    """
    # Example using albumentations
    # return A.Compose(
    #     [
    #         A.Resize(height=img_height, width=img_width),
    #         A.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ]
    # )
    return None
