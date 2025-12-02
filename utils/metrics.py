from sklearn import metrics


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
