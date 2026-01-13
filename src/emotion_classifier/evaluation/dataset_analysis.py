import torch as t
from emotion_classifier.utils import EMOTIONS



def find_label_cooccurance(dl, num_labels, device):
    cooccurance = t.zeros(num_labels, num_labels).to(device)
    for batch in dl:
        y = batch['labels'].float().to(device)  # shape: (B, c)
        cooccurance += y.T @ y                  # matmul (c, B) @ (B, c) -> (c, c)
    
    conditional_prob = cooccurance / (cooccurance.diag().unsqueeze(1) + 1e-8)
    diag = cooccurance.diag()
    jaccard_similarity = cooccurance / (diag.unsqueeze(1) + diag.unsqueeze(0) - cooccurance + 1e-8)
    cooccurance_matrices = {
        'raw-nums': cooccurance.cpu().numpy(),
        'conditional': conditional_prob.cpu().numpy(),
        'jaccard': jaccard_similarity.cpu().numpy()
    }
    return cooccurance_matrices


def count_labels(dl, num_labels, device):
    num_label_samples = t.zeros(num_labels).to(device)
    for batch in dl:
        y = batch["labels"].float().to(device)
        num_label_samples += y.sum(dim=0)
    
    return {
        EMOTIONS[i]: int(num_label_samples[i].cpu().numpy())
        for i in range(num_labels)
    }
