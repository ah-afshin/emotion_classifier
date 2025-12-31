import torch as t
from transformers import AutoTokenizer

from emotion_classifier.utils import EMOTIONS



def get_pred_probs(model, dl, device, method=None):
    probs, labels = [], []
    with t.no_grad():
        for batch in dl:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            if method:
                logit = model(x, mask, method)
            else:
                logit = model(x, mask)
            prob = t.sigmoid(logit)

            probs.append(prob)
            labels.append(y)
    return t.cat(probs), t.cat(labels)


def tokenize(text, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(text, truncation=True, max_length=128)
    input_ids = t.tensor(encoding["input_ids"]).unsqueeze(0)
    attention_mask = t.tensor(encoding["attention_mask"]).unsqueeze(0)
    return input_ids, attention_mask


def predict(model, input_ids, mask, thresholds, device):
    model.to(device)
    with t.no_grad():
        logit = model(input_ids.to(device), mask.to(device))
    prob = t.sigmoid(logit).cpu().numpy()
    pred = (prob >= thresholds).astype(int)
    
    emotions = {
        EMOTIONS[i]: prob[0][i]
        for i in range(28)
        if pred[0][i] == 1
    }
    return emotions
