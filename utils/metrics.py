from sklearn import metrics


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred, zero_division=0),
        'precision-micro': metrics.precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall-micro': metrics.recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1-micro': metrics.f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1-macro': metrics.f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1-samples': metrics.f1_score(y_true, y_pred, average='samples', zero_division=0),
        'hamming-loss': metrics.hamming_loss(y_true, y_pred) 
    }
