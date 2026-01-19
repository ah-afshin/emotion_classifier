import csv
from .text import EMOTIONS


def save_perclass_metrics_csv(perclass_metrics, fieldnames, path, filename):
    path.mkdir(parents=True, exist_ok=True)
    path = path/filename
    fieldnames = ["emotion"] + fieldnames
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for emotion, metrics in perclass_metrics.items():
            row = {"emotion": emotion}
            row.update(metrics)
            writer.writerow(row)


def save_cooccurrence_matrix_csv(matrix, path, filename):
    path.mkdir(parents=True, exist_ok=True)
    assert len(matrix) == len(EMOTIONS)

    path = path/filename
    fieldnames = ["emotion"] + EMOTIONS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(matrix)):
            row = {"emotion": EMOTIONS[i]}
            row.update({
                        EMOTIONS[j]:matrix[i][j]
                        for j in range(len(matrix[i]))
                    })
            writer.writerow(row)
