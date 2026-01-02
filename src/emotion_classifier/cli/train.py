import yaml, argparse
from pathlib import Path
from emotion_classifier.training import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path ,required=True)
    parser.add_argument("--output-dir", type=Path, required=False,
                        default=Path(__file__).parent.parent.parent.parent/'outputs')
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    with open(config_path) as f:
        config = yaml.safe_load(f)
    run_training(config, config_path, output_dir)


if __name__=="__main__":
    main()
