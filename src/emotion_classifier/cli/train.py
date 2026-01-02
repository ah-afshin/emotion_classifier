import yaml
from emotion_classifier.training import run_training


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    run_training(config)

if __name__=="__main__":
    main()
