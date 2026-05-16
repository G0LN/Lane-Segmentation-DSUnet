import torch
import numpy as np

def compute_confusion_matrix(preds, labels, num_classes):
    """
    Compute confusion matrix for a single batch.
    preds: [N, H, W] - Predicted class indices
    labels: [N, H, W] - Ground truth class indices
    """
    preds_flat = preds.cpu().numpy().flatten()
    labels_flat = labels.cpu().numpy().flatten()
    
    # Filter out-of-bounds labels if any
    mask = (labels_flat >= 0) & (labels_flat < num_classes)
    hist = np.bincount(
        num_classes * labels_flat[mask].astype(int) + preds_flat[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist

def get_metrics_from_conf_matrix(conf_matrix):
    """
    Compute mIoU, Accuracy, Precision, Recall, and F1 from a confusion matrix.
    conf_matrix: [num_classes, num_classes]
    """
    # True Positives (diagonal)
    tp = np.diag(conf_matrix)
    # Sum of rows (Ground Truth)
    gt_sum = conf_matrix.sum(axis=1)
    # Sum of columns (Predictions)
    pred_sum = conf_matrix.sum(axis=0)
    
    # Intersection = TP
    # Union = GT_sum + Pred_sum - TP
    union = gt_sum + pred_sum - tp
    
    # IoU per class
    iou = tp / np.maximum(union, 1e-6)
    miou = np.nanmean(iou)
    
    # Accuracy (Global)
    accuracy = tp.sum() / np.maximum(conf_matrix.sum(), 1e-6)
    
    # Precision per class: TP / Pred_sum
    precision = tp / np.maximum(pred_sum, 1e-6)
    mean_precision = np.nanmean(precision)
    
    # Recall per class: TP / GT_sum
    recall = tp / np.maximum(gt_sum, 1e-6)
    mean_recall = np.nanmean(recall)
    
    # F1 per class: 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-6)
    mean_f1 = np.nanmean(f1)
    
    return {
        "mIoU": float(miou),
        "Accuracy": float(accuracy),
        "Precision": float(mean_precision),
        "Recall": float(mean_recall),
        "F1": float(mean_f1),
        "ConfusionMatrix": conf_matrix
    }

def get_metrics(preds, labels, num_classes):
    """
    Legacy function optimized for memory.
    preds: Raw logits [N, C, H, W] or Predicted indices [N, H, W]
    labels: Ground truth [N, H, W]
    """
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)
        
    cm = compute_confusion_matrix(preds, labels, num_classes)
    return get_metrics_from_conf_matrix(cm)

