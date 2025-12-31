import json

from emotion_classifier.utils import setup_device, setup_path, save_perclass_csv
from emotion_classifier.data import get_dataloaders
from emotion_classifier.models import build_model
from emotion_classifier.inference import get_pred_probs
from .evaluator import test_predictions
from .threshold_tuner import find_global_threshold, find_perclass_threshold



def run_evaluation(config, thresholds, path):
    device = setup_device(config)
    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    tokenizer = config['data']['preprocessing']['tokenizer']
    _, _, test_dl = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    model = build_model(config, path)
    results, confusion = test_predictions(model, test_dl,  thresholds, device)

    setup_path(f'./outputs/{path}/eval/')
    with open(f'./outputs/{path}/eval/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    with open(f'./outputs/{path}/eval/confusion_stats.json', 'w', encoding='utf-8') as f:
        json.dump(confusion, f, indent=4)
    save_perclass_csv(
        results["per-class-metrics"],
        ["accuracy", "precision-micro", "recall-micro", "f1-micro", "f1-macro"],
        f"./outputs/{path}/eval/per_class_metrics_test.csv"
    )
    save_perclass_csv(
        confusion,
        ['true-negative', 'false-positive', 'false-negative', 'true-positive'],
        f"./outputs/{path}/eval/per_class_confusion_matrix.csv"
    )


def run_thresholding(config, path):
    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    device = setup_device(config)
    tokenizer = config['data']['preprocessing']['tokenizer']
    _, val_dl, _ = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    model = build_model(config, path)
    model.to(device).eval()
    pred_probs, true_labels  = get_pred_probs(model, val_dl, device, config['model']['variant'])
    
    thresholds = {}
    thresholds['global'], f1 = find_global_threshold(pred_probs, true_labels)
    thresholds['per_class'] = find_perclass_threshold(pred_probs, true_labels)

    with open(f'./outputs/{path}/thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=4)
    
    return thresholds['global'], f1
