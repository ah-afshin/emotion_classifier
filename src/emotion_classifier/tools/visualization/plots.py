# this code is AI generated
import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

from .helpers import _save



def plot_losses(data, out_dir: Path):
    data = sorted(data, key=lambda d: d["epoch"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([d["epoch"] for d in data], [d["train_loss"] for d in data], label="Train")
    ax.plot([d["epoch"] for d in data], [d["val_loss"] for d in data], label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True)

    _save(fig, out_dir / "loss_curve.png")



def plot_f1(data, metric: str, out_dir: Path):
    data = [d for d in data if metric in d]
    if not data:
        print(f"[WARN] metric '{metric}' not found")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([d["epoch"] for d in data], [d[metric] for d in data])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over epochs")
    ax.grid(True)

    _save(fig, out_dir / f"{metric}.png")


def plot_confusion_matrices(data, out_dir: Path, cols=4):
    rows = math.ceil(len(data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for ax, row in zip(axes, data):
        cm = np.array([
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]]
        ], dtype=float)

        # --- row-wise normalization (Actual-based) ---
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

        im = ax.imshow(
            cm_norm,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0
        )

        ax.set_title(row["emotion"], fontsize=10)

        # --- annotations ---
        for y in range(2):
            for x in range(2):
                value = int(cm[y, x])
                pct = cm_norm[y, x] * 100 if row_sums[y, 0] > 0 else 0

                color = "white" if cm_norm[y, x] > 0.5 else "black"

                ax.text(
                    x, y,
                    f"{value}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred -", "Pred +"])
        ax.set_yticklabels(["Actual -", "Actual +"])

    for ax in axes[len(data):]:
        ax.axis("off")

    # # --- shared colorbar ---
    # cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.6)
    # cbar.set_label("Fraction of actual class")

    fig.suptitle("Per-class Confusion Matrices (Row-normalized)", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    _save(fig, out_dir / "confusion_matrices.png")


def plot_heatmaps(title1, labels1, mat1,
                  title2, labels2, mat2,
                  out_path: Path):

    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    vmin = min(mat1.min(), mat2.min())
    vmax = max(mat1.max(), mat2.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    im1 = ax1.imshow(mat1, cmap="magma", vmin=vmin, vmax=vmax)
    ax1.set_title(title1)
    ax1.set_xticks(range(len(labels1)))
    ax1.set_yticks(range(len(labels1)))
    ax1.set_xticklabels(labels1, rotation=90)
    ax1.set_yticklabels(labels1)

    im2 = ax2.imshow(mat2, cmap="magma", vmin=vmin, vmax=vmax)
    ax2.set_title(title2)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_yticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, rotation=90)
    ax2.set_yticklabels(labels2)

    # fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8)
    fig.tight_layout()

    _save(fig, out_path)


def plot_json_barchart(json_file: Path, split: str, out_dir: Path):
    with open(json_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data.keys(), data.values())

    ax.set_xlabel("Emotions")
    ax.set_ylabel("Count")
    ax.set_title(f"Number of samples ({split})")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save(fig, out_dir / f"label_distribution_{split}.png")
