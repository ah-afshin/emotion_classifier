import torch as t
from emotion_classifier.utils import EMOTIONS



def find_label_cooccurance(dl, num_labels=28):
    cooccurance = t.zeros(num_labels, num_labels)
    for batch in dl:
        y = batch['labels'].float()     # shape: (B, c)
        cooccurance += y.T @ y          # matmul (c, B) @ (B, c) -> (c, c)
    
    conditional_prob = cooccurance / (cooccurance.diag().unsqueeze(1) + 1e-8)
    diag = cooccurance.diag()
    jaccard_similarity = cooccurance / (diag.unsqueeze(1) + diag.unsqueeze(0) - cooccurance + 1e-8)
    cooccurance_matrices = {
        'raw-nums': cooccurance.numpy(),
        'conditional': conditional_prob.numpy(),
        'jaccard': jaccard_similarity.numpy()
    }
    return cooccurance_matrices


def count_labels(dl, num_labels=28):
    num_label_samples = t.zeros(num_labels)
    for batch in dl:
        y = batch["labels"].float()
        num_label_samples += y.sum(dim=0)
    
    return {
        EMOTIONS[i]: num_label_samples[i].numpy()
        for i in range(num_labels)
    }
