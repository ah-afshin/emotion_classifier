# this code is AI generated
import json
import matplotlib.pyplot as plt


def plot_losses(data):
    epochs = [d['epoch'] for d in data]
    train_losses = [d['train_loss'] for d in data]
    val_losses = [d['val_loss'] for d in data]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_f1(data, metric='f1-macro'):
    epochs = [d['epoch'] for d in data]
    f1_scores = [d[metric] for d in data]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, f1_scores, label=f'{metric} F1 Score', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{metric} F1 Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_heatmaps(
    label1, row_headers1, column_headers1, matrix1,
    label2 ,row_headers2, column_headers2, matrix2
):
    """
    Plots two heatmaps side by side in one figure.
    
    Parameters:
    - row_headers1, column_headers1, matrix1: Data for the left heatmap.
    - row_headers2, column_headers2, matrix2: Data for the right heatmap.
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Adjust figure size as needed

    # Plot the first heatmap (left)
    im1 = ax1.imshow(matrix1, cmap='viridis', aspect='auto', origin='lower')
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label(label1)  # Or adjust as needed

    # Set x and y labels for the first heatmap
    ax1.set_xticks(range(len(column_headers1)))
    ax1.set_xticklabels(column_headers1, rotation=90)
    ax1.set_yticks(range(len(row_headers1)))
    ax1.set_yticklabels(row_headers1)
    ax1.set_title(label1)

    # Plot the second heatmap (right)
    im2 = ax2.imshow(matrix2, cmap='viridis', aspect='auto', origin='lower')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label(label2)  # Or adjust as needed

    # Set x and y labels for the second heatmap
    ax2.set_xticks(range(len(column_headers2)))
    ax2.set_xticklabels(column_headers2, rotation=90)
    ax2.set_yticks(range(len(row_headers2)))
    ax2.set_yticklabels(row_headers2)
    ax2.set_title(label2)

    # Improve layout
    fig.tight_layout()
    plt.show()


def plot_confusion_matrices(data, title="Per-class Confusion Matrices", figsize=(20, 20)):
    """
    Plot confusion matrices for each class in a grid of subplots.

    Parameters:
        - data: List of dictionaries with confusion matrix data.
        - title: Title of the figure.
        - figsize: Figure size.
    """
    num_classes = len(data)
    num_rows = 7
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= num_classes:
            ax.axis('off')  # Hide unused subplots
            continue

        # Extract confusion matrix data for this class
        row = data[i]
        tn = row['true-negative']
        fp = row['false-positive']
        fn = row['false-negative']
        tp = row['true-positive']

        # Create the confusion matrix (2x2)
        cm = [[tn, fp], [fn, tp]]

        # Plot the confusion matrix
        im = ax.imshow(cm, cmap='viridis', aspect='auto')
        ax.set_title(row['emotion'])

        # Annotate the cells
        for j in range(2):
            for k in range(2):
                ax.text(k, j, f"{cm[j][k]}", ha="center", va="center", color="black")

        # Set axis labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Negative', 'Positive'])

        # Set axis labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_json_barchart(json_file_path, split):
    """
    Loads data from a JSON file and plots it as a bar chart.

    Args:
        json_file_path (str): The path to the JSON file.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    emotions = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
    plt.bar(emotions, values)
    plt.xlabel("Emotions")
    plt.ylabel("Values")
    plt.title(f"Number of samples for each label ({split} dataset)")
    plt.xticks(rotation=45, ha="right") # Rotate x-axis labels for better readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
