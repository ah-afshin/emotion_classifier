import torch as t
import numpy as np
from utils.metrics import compute_metrics, compute_confusion_like, compute_prediction_metrics, compute_binary_metrics
from utils.text import EMOTIONS



def compute_metrics_perclass(true_labels, pred_labels):
    metrics = {}
    for c in range(28):     # we have 28 classes
        y_pred = pred_labels[:, c]
        y_true = true_labels[:, c]
        metrics[EMOTIONS[c]] = compute_binary_metrics(y_true, y_pred)
    return metrics
    

def test_predictions(model, dl, thresholds, device, method=None):
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    model.eval().to(device)
    with t.no_grad():
        for batch in dl:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            if method:
                logit = model(x, mask, method)
            else:
                logit = model(x, mask)
            
            prob = t.sigmoid(logit).cpu().numpy()
            pred = (prob >= thresholds).astype(int)
            y_true_list.append(y.cpu().numpy().astype(int))
            y_prob_list.append(prob)
            y_pred_list.append(pred)
        
        y_true = np.concatenate(y_true_list, axis=0)        # shape: (n_samples, num_labels)
        y_prob = np.concatenate(y_prob_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
    
    global_metrics = compute_metrics(y_true, y_pred)
    perclass_metrics = compute_metrics_perclass(y_true, y_pred)
    decisioning_metrics = compute_prediction_metrics(y_true, y_prob)
    confusion_like_matrix = compute_confusion_like(y_true, y_pred)
    return {'metrics': global_metrics, 'per-class-metrics': perclass_metrics, 'prediction-stats': decisioning_metrics}, confusion_like_matrix
