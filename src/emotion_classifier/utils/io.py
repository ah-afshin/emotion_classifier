import csv


def save_csv(perclass_metrics, fieldnames, path, filename):
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
