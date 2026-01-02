import yaml, json
from sys import argv
import numpy as np
from emotion_classifier.models import build_model
from emotion_classifier.utils import setup_device
from emotion_classifier.inference import predict, tokenize


def main():
    with open(f'./outputs/{argv[1]}/config.yaml') as f:
        config = yaml.safe_load(f)
    with open(f'./outputs/{argv[1]}/thresholds.json') as f:
        thresholds = json.load(f)
        thresholds = np.array(thresholds['per_class'])
    
    tokenizer = config['data']['preprocessing']['tokenizer']
    text, mask = tokenize(argv[2], tokenizer)
    
    model = build_model(config, argv[1])
    model.eval()
    results = predict(model, text, mask, thresholds)
    
    print(results)


if __name__=="__main__":
    main()
