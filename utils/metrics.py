from sklearn import metrics
from utils.text import EMOTIONS


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(metrics.accuracy_score(y_true, y_pred)),
        'precision-micro': float(metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'recall-micro': float(metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1-micro': float(metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1-macro': float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1-samples': float(metrics.f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'hamming-loss': float(metrics.hamming_loss(y_true, y_pred)) 
    }


def compute_binary_metrics(y_true, y_pred):
    return {
        'accuracy': float(metrics.accuracy_score(y_true, y_pred)),
        'precision-micro': float(metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'recall-micro': float(metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1-micro': float(metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'f1-macro': float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0))
    }


def compute_prediction_metrics(true_labels, pred_prob):
    try:
        roc_auc_score = metrics.roc_auc_score(true_labels, pred_prob, average='macro')
    except:
        roc_auc_score = None
    return {
        'ROC-AUC': roc_auc_score,
        'positive-rate': float(pred_prob.mean()),
        'coverage': {
            EMOTIONS[c]: float(pred_prob[:, c].mean())
            for c in range(28)
        }
    }


def compute_confusion_like(y_true, y_pred):
    mcm = metrics.multilabel_confusion_matrix(y_true, y_pred)
    confusion_like = {}
    for c in range(28):
        tn, fp, fn, tp = mcm[c].ravel()
        confusion_like[EMOTIONS[c]] = {
                    'true-negative': int(tn),
                    'false-positive': int(fp),
                    'false-negative': int(fn),
                    'true-positive': int(tp)
            }
    return confusion_like
