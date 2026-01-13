import torch as t
import numpy as np
from emotion_classifier.utils.metrics import compute_metrics, compute_confusion_like, compute_prediction_metrics, compute_binary_metrics
from emotion_classifier.utils.text import EMOTIONS



def compute_metrics_perclass(true_labels, pred_labels):
    metrics = {}
    for c in range(28):     # we have 28 classes
        y_pred = pred_labels[:, c]
        y_true = true_labels[:, c]
        metrics[EMOTIONS[c]] = compute_binary_metrics(y_true, y_pred)
    return metrics
    

def compute_cooccurance(y_pred, y_true):
    C = y_pred.T @ y_true       # (c, N) @ (N, c)
    diag = C.diag()
    cooccurance_matrices = {
        'raw_nums': C.numpy(),
        'conditional': (C / (C.diag().unsqueeze(1) + 1e-8)).numpy(),
        'jaccard': (C / (diag.unsqueeze(1) + diag.unsqueeze(0) - C + 1e-8)).numpy()
    }
    return cooccurance_matrices


def diagnose_model(y_pred, y_true):
    y_pred = t.tensor(y_pred).float()
    y_true = t.tensor(y_true).float()
    y_false = (y_pred-y_true).absolute()

    pred_true_cooccurance = compute_cooccurance(y_pred, y_true)
    false_true_cooccurance = compute_cooccurance(y_false, y_true)
    return pred_true_cooccurance, false_true_cooccurance


def test_predictions(model, dl, thresholds, device):
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    model.eval().to(device)
    with t.no_grad():
        for batch in dl:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

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
    pred_cooccure, false_cooccure = diagnose_model(y_pred, y_true)

    return {
            'metrics': global_metrics,
            'per-class-metrics': perclass_metrics,
            'prediction-stats': decisioning_metrics
        },{
            'confusion-like': confusion_like_matrix,
            'pred-true-cooccure': pred_cooccure,
            'false-true-cooccure': false_cooccure
        }
