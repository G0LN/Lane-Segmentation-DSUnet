import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(image, mask, pred, save_path=None):
    """
    Helper function to visualize image, ground truth mask, and prediction
    image: tensor or numpy array [C, H, W] or [H, W, C]
    mask: tensor or numpy array [H, W]
    pred: tensor or numpy array [H, W]
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Process image if it's a tensor
    if torch.is_tensor(image):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image
        
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
        
    if torch.is_tensor(pred):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred
        
    axs[0].imshow(img_np)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    
    axs[1].imshow(mask_np)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')
    
    axs[2].imshow(pred_np)
    axs[2].set_title("Prediction")
    axs[2].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
