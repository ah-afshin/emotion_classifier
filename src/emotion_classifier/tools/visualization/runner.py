from pathlib import Path
from .loaders import parse_log, parse_confusion_matrix, read_similarity_matrix
from .plots import plot_confusion_matrices, plot_f1, plot_heatmaps, plot_losses, plot_json_barchart



def run_visualization(path: Path):
    path = Path(path)
    out_dir = path / "plots"

    if "dataset" in str(path):
        for split in ["train", "eval", "test"]:
            plot_json_barchart(
                path / split / "num_label_samples.json",
                split,
                out_dir
            )

        for name in ["matrix_raw", "conditional_probability", "jaccard_similarity"]:
            labels_tr, mat_tr = read_similarity_matrix(path / "train" / f"cooccurrence_{name}.csv")
            labels_te, mat_te = read_similarity_matrix(path / "test" / f"cooccurrence_{name}.csv")

            plot_heatmaps(
                f"Train - {name}", labels_tr, mat_tr,
                f"Test - {name}", labels_te, mat_te,
                out_dir / f"cooccurrence_{name}.png"
            )

    else:
        logs = parse_log((path / "train.log").read_text().splitlines())

        plot_losses(logs, out_dir)
        plot_f1(logs, "f1-micro", out_dir)

        cms = parse_confusion_matrix(path / "eval" / "per_class_confusion_matrix.csv")
        plot_confusion_matrices(cms, out_dir)
        
        try:
            for name in ["raw_nums", "conditional", "jaccard"]:
                labels_p, mat_p = read_similarity_matrix(
                    path / "eval" / f"pred_true_cooccurrence_matrix_{name}.csv"
                )
                labels_e, mat_e = read_similarity_matrix(
                    path / "eval" / f"false_true_cooccurrence_matrix_{name}.csv"
                )

                plot_heatmaps(
                    f"Prediction - {name}", labels_p, mat_p,
                    f"Error - {name}", labels_e, mat_e,
                    out_dir / f"error_analysis_{name}.png"
                )
        except:
            print('no cooccurrence matrices found.')
