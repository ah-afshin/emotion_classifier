import yaml
import json
import torch as t
import numpy as np
from sys import argv
from transformers import AutoTokenizer

from helpers import setup_device
from utils.text import EMOTIONS
from utils.text import get_vocab_size
from models.bilstm import EmotionClassifierBiLSTM
from models.transformer import EmotionClassifierTransformer


def tokenize(text, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(text, truncation=True, max_length=128)
    input_ids = t.tensor(encoding["input_ids"]).unsqueeze(0)
    attention_mask = t.tensor(encoding["attention_mask"]).unsqueeze(0)
    return input_ids, attention_mask


def predict(model, input_ids, mask, thresholds, method=None):
    model.to(device)
    with t.no_grad():
        if method:
            logit = model(input_ids.to(device), mask.to(device), method)
        else:
            logit = model(input_ids.to(device), mask.to(device))
    prob = t.sigmoid(logit).cpu().numpy()
    pred = (prob >= thresholds).astype(int)
    
    emotions = {
        EMOTIONS[i]: prob[0][i]
        for i in range(28)
        if pred[0][i] == 1
    }
    return emotions


if __name__ == "__main__":
    with open(f'./outputs/{argv[1]}/config.yaml') as f:
        config = yaml.safe_load(f)
    with open(f'./outputs/{argv[1]}/thresholds.json') as f:
        thresholds = json.load(f)
        thresholds = np.array(thresholds['per_class'])

    device = setup_device(config)
    tokenizer = config['data']['preprocessing']['tokenizer']
    text, mask = tokenize(argv[2], tokenizer)
    
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
            results = predict(model, text, mask, thresholds, config['model']['variant'])

        case 'transformer':
            model = EmotionClassifierTransformer(
                            mode=config['model']['variant'],
                            model_name=config['model']['transformer']['transformer_model'],
                            dropout=config['model']['transformer']['dropout']
                    )
            model.load_state_dict(t.load(f'./outputs/{argv[1]}/best_model.pt', weights_only=True))
            model.eval()
            results = predict(model, text, mask, thresholds)
    
    print(results)
