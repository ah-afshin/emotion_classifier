# this code is AI generated
import csv, re



def parse_confusion_matrix(csv_file):
    """
    Parse a CSV file containing confusion matrix data.

    Parameters:
        - csv_file: Path to the CSV file.

    Returns:
        - A list of dictionaries with keys: 'emotion', 'true-negative', 'false-positive', 'false-negative', 'true-positive'.
    """
    data = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'emotion': row['emotion'],
                'true-negative': float(row['true-negative']),
                'false-positive': float(row['false-positive']),
                'false-negative': float(row['false-negative']),
                'true-positive': float(row['true-positive']),
            })
    return data


def read_similarity_matrix(file_path):
    """Read a CSV file and return a 2D list of similarity values and emotion names."""
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    # Extract column headers (first row)
    column_headers = data[0]

    # Extract row headers (first column of each row after the first)
    row_headers = [row[0] for row in data[1:]]

    # Extract the similarity matrix (27x27)
    matrix = []
    for row in data[1:]:
        matrix_row = [float(val) for val in row[1:]]
        matrix.append(matrix_row)

    return row_headers, column_headers, matrix


def parse_log(log_lines):
    data = []
    i = 0
    while i < len(log_lines):
        line = log_lines[i].strip()
        if line.startswith('epoch'):
            # Parse epoch, train_loss, validation_loss
            parts = line.split('|')
            epoch_part = parts[0].split(' ')[1]
            epoch = int(epoch_part)
            train_loss = float(parts[1].split(':')[1].strip())
            val_loss = float(parts[2].split(':')[1].strip())

            # Find the metrics line
            i += 1
            while i < len(log_lines):
                metrics_line = log_lines[i].strip()
                if metrics_line.startswith('Metrics:'):
                    # Parse metrics
                    metrics = {}
                    metrics_str = metrics_line[len('Metrics: '):]
                    metrics_items = re.findall(r'(\w+(-\w+)*)\s*:\s*([-+]?\d*\.?\d+)', metrics_str)
                    for key, _, value in metrics_items:
                        metrics[key] = float(value)

                    # Append to data
                    data.append({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **metrics
                    })
                    i += 1
                    break
                else:
                    i += 1
        else:
            i += 1
    return data
