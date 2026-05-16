import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(history, save_dir):
    """
    Plots Loss, mIoU, Precision, Recall, F1 curves from history dictionary.
    history = {
        'train_loss': [], 'val_loss': [],
        'val_miou': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': []
    }
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. mIoU Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_miou'], label='Validation mIoU', color='orange')
    plt.title('mIoU Curve')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'miou_curve.png'))
    plt.close()
    
    # 3. Precision, Recall, F1 Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_precision'], label='Precision')
    plt.plot(epochs, history['val_recall'], label='Recall')
    plt.plot(epochs, history['val_f1'], label='F1 Score')
    plt.title('Precision, Recall, F1 Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pr_f1_curve.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, save_dir, epoch=None):
    """
    Plots a Confusion Matrix using seaborn heatmap.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # handle division by zero
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    title = 'Normalized Confusion Matrix'
    if epoch is not None:
        title += f' (Epoch {epoch})'
        
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_epoch_{epoch}.png' if epoch else 'confusion_matrix.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
