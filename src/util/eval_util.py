import numpy as np
from sklearn.metrics import roc_curve
from src.models.base_model import BaseModel
import torch
from torch.utils.data import DataLoader


def evaluate_model(model: BaseModel, dl: DataLoader, n_batches: int):
    all_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for _ in range(n_batches):
            batch = next(iter(dl))
            out = model.forward(batch.cuda())
            l = model.compute_loss(out, batch)
            if l.metrics is not None:
                all_metrics.append(l.metrics)
    # Aggregate metrics over all batches
    aggregated_metrics = {}
    for metrics in all_metrics:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)
    # Compute mean of each metric
    for key in aggregated_metrics:
        aggregated_metrics[key] = sum(aggregated_metrics[key]) / len(
            aggregated_metrics[key]
        )
    return aggregated_metrics


def get_optimal_threshold_auc(y_true: torch.Tensor, y_score: torch.Tensor):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def get_optimal_threshold_iou(y_true: np.ndarray, y_score: np.ndarray):
    best_iou = 0
    best_threshold = 0
    for threshold in np.linspace(0, 1, 100):
        pred_masks = y_score > threshold
        _, iou = get_dice_ji(pred_masks, y_true)
        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold
    return best_threshold


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji
