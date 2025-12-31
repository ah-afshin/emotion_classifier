import yaml
from sys import argv
from emotion_classifier.evaluation import run_thresholding


if __name__=="__main__":
    with open(f'./outputs/{argv[1]}/config.yaml') as f:
        config = yaml.safe_load(f)
    threshold, f1 = run_thresholding(config, argv[1])

    print(f'Best global threshold: {threshold}')
    print(f'F1-score for global threshold: {f1}')
