import yaml
from emotion_classifier.training import run_training


if __name__=="__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    run_training(config)
