import yaml, json, argparse
from pathlib import Path
import numpy as np
from emotion_classifier.models import build_model
from emotion_classifier.inference import predict, tokenize
from emotion_classifier.utils import setup_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    path = args.model_path.expanduser().resolve()
    text = args.text

    with open(path/'config.yaml') as f:
        config = yaml.safe_load(f)
    with open(path/'thresholds.json') as f:
        thresholds = json.load(f)
        thresholds = np.array(thresholds['per_class'])
    
    tokenizer = config['data']['preprocessing']['tokenizer']
    text, mask = tokenize(text, tokenizer)
    device = setup_device(config)
    
    model = build_model(config, path)
    model.eval()
    results = predict(model, text, mask, thresholds, device)
    
    print(results)


if __name__=="__main__":
    main()
