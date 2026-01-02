import yaml, argparse
from pathlib import Path
from emotion_classifier.evaluation import run_thresholding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    path = args.model_path.expanduser().resolve()

    with open(path/'config.yaml') as f:
        config = yaml.safe_load(f)
    threshold, f1 = run_thresholding(config, path)

    print(f'Best global threshold: {threshold}')
    print(f'F1-score for global threshold: {f1}')


if __name__=="__main__":
    main()
