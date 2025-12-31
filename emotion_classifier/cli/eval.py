import yaml, json
from sys import argv
import numpy as np
from emotion_classifier.evaluation import run_evaluation


if __name__=="__main__":
    with open(f'./outputs/{argv[1]}/config.yaml') as f:
        config = yaml.safe_load(f)
    with open(f'./outputs/{argv[1]}/thresholds.json') as f:
        thresholds = json.load(f)
        thresholds = np.array(thresholds['per_class'])
    
    run_evaluation(config, thresholds, argv[1])
