import numpy as np
from sklearn import metrics


def find_global_threshold(pred_probs, labels, thresholds=np.linspace(0.05, 0.5, 50)):
    y_true = labels.cpu().numpy()
    pred_probs = pred_probs.cpu().numpy()
    best_metric, best_threshold = -1, None
    
    for threshold in thresholds:
        y_pred = (pred_probs >= threshold).astype(int)
        metric = float(metrics.f1_score(y_true, y_pred, average='micro', zero_division=0))
        
        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold
    return best_threshold, best_metric


def find_perclass_threshold(pred_probs, labels, thresholds=np.linspace(0.05, 0.5, 50)):
    y_true = labels.cpu().numpy()
    pred_probs = pred_probs.cpu().numpy()
    best_thresholds = []

    for c in range(28):     # we have 28 classes
        best_m, best_t = -1, None
        for threshold in thresholds:
            y_pred = (pred_probs[:, c] >= threshold).astype(int)
            metric = float(metrics.f1_score(y_true[:, c], y_pred, zero_division=0))
            if metric > best_m:
                best_t = threshold
                best_m = metric
        best_thresholds.append(best_t)
    return best_thresholds
