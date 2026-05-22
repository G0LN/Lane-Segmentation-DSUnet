import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(history, save_dir):
    """
    Plots Loss, mIoU, Precision & Recall (combined), and F1 Score (separated) curves.
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
    
    # 3. Precision & Recall Curves (F1 score is now separated!)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_precision'], label='Precision', color='blue')
    plt.plot(epochs, history['val_recall'], label='Recall', color='green')
    plt.title('Precision & Recall Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()

    # 4. F1 Score Curve (Separate metric!)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], label='F1 Score', color='purple')
    plt.title('F1 Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, save_dir, epoch=None):
    """
    Plots and saves both raw and normalized Confusion Matrices.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. Raw Confusion Matrix (Normal form with integers)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    title = 'Confusion Matrix'
    if epoch is not None:
        title += f' (Epoch {epoch})'
        
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_epoch_{epoch}.png' if epoch else 'confusion_matrix.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    # 2. Normalized Confusion Matrix (Normalized form with floats)
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-6)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    title_norm = 'Normalized Confusion Matrix'
    if epoch is not None:
        title_norm += f' (Epoch {epoch})'
        
    plt.title(title_norm)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename_norm = f'confusion_matrix_normalized_epoch_{epoch}.png' if epoch else 'confusion_matrix_normalized.png'
    plt.savefig(os.path.join(save_dir, filename_norm))
    plt.close()

def plot_precision_recall_curve(precision, recall, class_names, save_dir):
    """
    Plots the Precision-Recall curve for all classes.
    precision: [num_classes, num_thresholds]
    recall: [num_classes, num_thresholds]
    class_names: List of class names
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 8))
    for c in range(len(class_names)):
        plt.plot(recall[c], precision[c], marker='.', label=class_names[c])
        
    plt.title('Precision-Recall Curve (per Class)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()
