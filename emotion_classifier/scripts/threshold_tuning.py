import yaml
import json
from sys import argv
import numpy as np
import torch as t
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


def get_pred_probs(model, dl, device, method=None):
    probs, labels = [], []
    with t.no_grad():
        for batch in dl:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            if method:
                logit = model(x, mask, method)
            else:
                logit = model(x, mask)
            prob = t.sigmoid(logit)

            probs.append(prob)
            labels.append(y)
    return t.cat(probs), t.cat(labels)


if __name__ == "__main__":
    from helpers import setup_device
    from utils.text import get_vocab_size
    from data.preprocess import get_dataloaders
    from models.bilstm import EmotionClassifierBiLSTM
    from models.transformer import EmotionClassifierTransformer

    with open(f'./outputs/{argv[1]}/config.yaml') as f:
        config = yaml.safe_load(f)
    
    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    device = setup_device(config)
    tokenizer = config['data']['preprocessing']['tokenizer']
    _, val_dl, _ = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    match config['model']['name']:
        case 'bilstm':
            model = EmotionClassifierBiLSTM(
                            vocab_size=get_vocab_size(),
                            hidden_size=config['model']['bilstm']['hidden_size'],
                            num_layers=config['model']['bilstm']['num_layers'],
                            dropout=config['model']['bilstm']['dropout']
                    )
            model.load_state_dict(t.load(f'./outputs/{argv[1]}/best_model.pt', weights_only=True))
            model.to(device).eval()
            pred_probs, true_labels  = get_pred_probs(model, val_dl, device, config['model']['variant'])

        case 'transformer':
            model = EmotionClassifierTransformer(
                            mode=config['model']['variant'],
                            model_name=config['model']['transformer']['transformer_model'],
                            dropout=config['model']['transformer']['dropout']
                    )
            model.load_state_dict(t.load(f'./outputs/{argv[1]}/best_model.pt', weights_only=True))
            model.to(device).eval()
            pred_probs, true_labels = get_pred_probs(model, val_dl, device)
    
    thresholds = {}
    thresholds['global'], f1 = find_global_threshold(pred_probs, true_labels)
    thresholds['per_class'] = find_perclass_threshold(pred_probs, true_labels)

    with open(f'./outputs/{argv[1]}/thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=4)
    print(json.dumps(thresholds, indent=4))
    print(f'F1-score for global threshold: {f1}')
