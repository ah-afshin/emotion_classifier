import argparse
from pathlib import Path
from emotion_classifier.evaluation import run_dataset_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str ,required=True)
    parser.add_argument("--output-dir", type=Path, required=False,
                        default=Path(__file__).parent.parent.parent.parent/'outputs'/'dataset')
    parser.add_argument("--device", type=str, required=False, default='auto')

    args = parser.parse_args()
    split = args.split
    device = args.device
    output_dir = args.output_dir.expanduser().resolve()

    run_dataset_analysis(split, output_dir, device)


if __name__=="__main__":
    main()
