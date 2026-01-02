import yaml, json, argparse
from pathlib import Path
import numpy as np
from emotion_classifier.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    path = args.model_path.expanduser().resolve()

    with open(path/'config.yaml') as f:
        config = yaml.safe_load(f)
    with open(path/'thresholds.json') as f:
        thresholds = json.load(f)
        thresholds = np.array(thresholds['per_class'])
    
    run_evaluation(config, thresholds, path)


if __name__=="__main__":
    main()
