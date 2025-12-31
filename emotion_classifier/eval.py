import yaml
import json
from sys import argv
import torch as t
import numpy as np
from torch import nn
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



if __name__ == "__main__":
    from helpers import setup_device, setup_path
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    from utils.io import save_perclass_csv
    from models.bilstm import EmotionClassifierBiLSTM
    from models.transformer import EmotionClassifierTransformer


    with open(f'./outputs/{argv[1]}/config.yaml') as f:
        config = yaml.safe_load(f)
    with open(f'./outputs/{argv[1]}/thresholds.json') as f:
        thresholds = json.load(f)
        thresholds = np.array(thresholds['per_class'])
    results = {}

    device = setup_device(config)
    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    tokenizer = config['data']['preprocessing']['tokenizer']
    _, _, test_dl = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    match config['model']['name']:
        case 'bilstm':
            model = EmotionClassifierBiLSTM(
                            vocab_size=get_vocab_size(),
                            hidden_size=config['model']['bilstm']['hidden_size'],
                            num_layers=config['model']['bilstm']['num_layers'],
                            dropout=config['model']['bilstm']['dropout']
                    )
            model.load_state_dict(t.load(f'./outputs/{argv[1]}/best_model.pt', weights_only=True))
            model.eval()
            results, confusion = test_predictions(model, test_dl,  thresholds, device, config['model']['variant'])

        case 'transformer':
            model = EmotionClassifierTransformer(
                            mode=config['model']['variant'],
                            model_name=config['model']['transformer']['transformer_model'],
                            dropout=config['model']['transformer']['dropout']
                    )
            model.load_state_dict(t.load(f'./outputs/{argv[1]}/best_model.pt', weights_only=True))
            model.eval()
            results, confusion = test_predictions(model, test_dl,  thresholds, device)
    
    setup_path(f'outputs/{argv[1]}/eval/')
    with open(f'./outputs/{argv[1]}/eval/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    with open(f'./outputs/{argv[1]}/eval/confusion_stats.json', 'w', encoding='utf-8') as f:
        json.dump(confusion, f, indent=4)
    save_perclass_csv(
        results["per-class-metrics"],
        ["accuracy", "precision-micro", "recall-micro", "f1-micro", "f1-macro"],
        f"./outputs/{argv[1]}/eval/per_class_metrics_test.csv"
    )
    save_perclass_csv(
        confusion,
        ['true-negative', 'false-positive', 'false-negative', 'true-positive'],
        f"./outputs/{argv[1]}/eval/per_class_confusion_matrix.csv"
    )
    print(json.dumps(results, indent=4))
