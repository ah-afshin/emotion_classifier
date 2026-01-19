from .loaders import parse_log, parse_confusion_matrix, read_similarity_matrix
from .plots import plot_confusion_matrices, plot_f1, plot_heatmaps, plot_losses, plot_json_barchart



def run_visualization(path):
    if 'dataset' in str(path):
        plot_json_barchart(path/'train'/'num_label_samples.json', 'train')
        plot_json_barchart(path/'eval'/'num_label_samples.json', 'eval')
        plot_json_barchart(path/'test'/'num_label_samples.json', 'test')
        
        for i in {'matrix_raw', 'conditional_probability', 'jaccard_similarity'}:
            train_r_header, train_c_header, train_data = read_similarity_matrix(path/'train'/f'cooccurrence_{i}.csv')
            test_r_header, test_c_header, test_data = read_similarity_matrix(path/'test'/f'cooccurrence_{i}.csv')
            plot_heatmaps(
                'train dataset label cooccurrence', train_r_header, train_c_header, train_data,
                'test dataset label cooccurrence', test_r_header, test_c_header, test_data
            )

    else:
        with open(path/'train.log', 'r') as f:
            log_lines = f.readlines()
        logs = parse_log(log_lines)
        plot_losses(logs)
        plot_f1(logs, 'f1-micro')
        
        csv_path = path/'eval'/'per_class_confusion_matrix.csv'
        matrices = parse_confusion_matrix(csv_path)
        plot_confusion_matrices(matrices)

        for i in {'raw_nums', 'conditional', 'jaccard'}:
            pred_r_header, pred_c_header, pred_data = read_similarity_matrix(path/'eval'/f'pred_true_cooccurrence_matrix_{i}.csv')
            err_r_header, err_c_header, err_data = read_similarity_matrix(path/'eval'/f'false_true_cooccurrence_matrix_{i}.csv')
            plot_heatmaps(
                "prediction/label cooccurrence heatmap", pred_r_header, pred_c_header, pred_data,
                "error/label cooccurrence heatmap", err_r_header, err_c_header, err_data
            )
