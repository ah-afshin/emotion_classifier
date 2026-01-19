import yaml, argparse
from pathlib import Path
from emotion_classifier.tools.visualization import run_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path ,required=True)
    args = parser.parse_args()
    path = args.path.expanduser().resolve()

    run_visualization(path)


if __name__=="__main__":
    main()
