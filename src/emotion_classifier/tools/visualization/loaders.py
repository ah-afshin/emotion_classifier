# this code is AI generated
import csv
import re
from pathlib import Path



def parse_confusion_matrix(csv_file: Path):
    """
    Expected CSV columns:
    emotion, tn, fp, fn, tp
    """
    data = []
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "emotion": row["emotion"],
                "tn": float(row["true-negative"]),
                "fp": float(row["false-positive"]),
                "fn": float(row["false-negative"]),
                "tp": float(row["true-positive"]),
            })
    return data


def read_similarity_matrix(csv_file: Path):
    """
    CSV format:
        ,label1,label2,...
        label1,0.0,0.2,...
        label2,0.2,0.0,...
    """
    with open(csv_file, encoding="utf-8") as f:
        reader = list(csv.reader(f))

    headers = reader[0][1:]
    rows = [r[0] for r in reader[1:]]
    matrix = [[float(v) for v in r[1:]] for r in reader[1:]]

    if headers != rows:
        raise ValueError(f"Row/column labels mismatch in {csv_file}")

    return headers, matrix


def parse_log(lines):
    epoch_re = re.compile(
        r"epoch\s+(\d+)\s+\|\s+train_loss:\s*([\d.]+)\s+\|\s+validation_loss:\s*([\d.]+)"
    )
    metric_re = re.compile(r"([\w\-]+)\s*:\s*([-+]?\d*\.?\d+)")

    results = []
    i = 0

    while i < len(lines):
        m = epoch_re.search(lines[i])
        if not m:
            i += 1
            continue

        entry = {
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "val_loss": float(m.group(3)),
        }

        i += 1
        while i < len(lines) and "Metrics:" not in lines[i]:
            i += 1

        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if not line or "epoch" in line:
                break

            m = metric_re.match(line)
            if m:
                entry[m.group(1)] = float(m.group(2))
            i += 1

        results.append(entry)

    return results
