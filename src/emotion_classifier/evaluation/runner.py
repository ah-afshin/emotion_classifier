import json

from emotion_classifier.utils import setup_device, save_csv, EMOTIONS
from emotion_classifier.data import get_dataloaders
from emotion_classifier.models import build_model
from emotion_classifier.inference import get_pred_probs
from .evaluator import test_predictions
from .threshold_tuner import find_global_threshold, find_perclass_threshold
from .dataset_analysis import find_label_cooccurance, count_labels



def run_evaluation(config, thresholds, path):
    device = setup_device(config)
    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    tokenizer = config['data']['preprocessing']['tokenizer']
    _, _, test_dl = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    model = build_model(config, path)
    results, confusion = test_predictions(model, test_dl,  thresholds, device)

    (path/'eval').mkdir(parents=True, exist_ok=True)
    
    with open(path/'eval'/'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    with open(path/'eval'/'confusion_stats.json', 'w', encoding='utf-8') as f:
        json.dump(confusion, f, indent=4)
    
    save_csv(
        results["per-class-metrics"],
        ["accuracy", "precision-micro", "recall-micro", "f1-micro", "f1-macro"],
        path/"eval",
        "per_class_metrics_test.csv"
    )
    
    save_csv(
        confusion,
        ['true-negative', 'false-positive', 'false-negative', 'true-positive'],
        path/"eval",
        "per_class_confusion_matrix.csv"
    )


def run_thresholding(config, path):
    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    device = setup_device(config)
    tokenizer = config['data']['preprocessing']['tokenizer']
    _, val_dl, _ = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    model = build_model(config, path)
    model.to(device).eval()
    pred_probs, true_labels  = get_pred_probs(model, val_dl, device)
    
    thresholds = {}
    thresholds['global'], f1 = find_global_threshold(pred_probs, true_labels)
    thresholds['per_class'] = find_perclass_threshold(pred_probs, true_labels)

    with open(path/'thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=4)
    
    return thresholds['global'], f1


def run_dataset_analysis(split, path):
    (path/split).mkdir(parents=True, exist_ok=True)
    dls={}
    dls['train'], dls['eval'], dls['test'] = get_dataloaders()
    dl = dls[split]

    cooccurance = find_label_cooccurance(dl)
    save_csv(
        cooccurance['raw-nums'],
        EMOTIONS,
        path=path/split/'cooccurance_matrix_raw.csv'
    )
    save_csv(
        cooccurance['conditional'],
        EMOTIONS,
        path=path/split/'cooccurance_conditional_probability.csv'
    )
    save_csv(
        cooccurance['jaccard'],
        EMOTIONS,
        path=path/split/'cooccurance_jaccard_similarity.csv'
    )

    num_labels = count_labels(dl)
    with open(path/split/'num_label_samples.json', 'w', encoding='utf-8') as f:
        json.dump(num_labels, f, indent=4)
