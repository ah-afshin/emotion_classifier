from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

from utils.text import label2id, id2label, EMOTIONS  # هنوز باید ساخته شن



class GoEmotionsDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        self.dataset = load_dataset("go_emotions", "simplified", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        labels = item["labels"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # represent label as one-hot (multi-label classification)
        label_tensor = torch.zeros(len(label2id))
        for label_id in labels:
            label_tensor[label_id] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }


def get_dataloaders(tokenizer_name="bert-base-uncased", batch_size=32, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = GoEmotionsDataset("train", tokenizer, max_length)
    val_dataset   = GoEmotionsDataset("validation", tokenizer, max_length)
    test_dataset  = GoEmotionsDataset("test", tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
