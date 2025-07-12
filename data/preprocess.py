import os
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch as t

from utils.text import label2id



CACHE_DIR = "./data/processed"
os.makedirs(CACHE_DIR, exist_ok=True)


def preprocess_and_save(split, tokenizer, max_length=128):
    cache_path = os.path.join(CACHE_DIR, f"{split}_data.pt")

    if os.path.exists(cache_path):
        print(f"[+] Loaded cached {split} data.")
        return t.load(cache_path)

    print(f"[!] Preprocessing {split} data...")

    dataset = load_dataset("go_emotions", "simplified", split=split)
    input_ids_list = []
    attention_masks_list = []
    labels_list = []

    for item in dataset:
        text = item["text"]
        labels = item["labels"]

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        label_tensor = t.zeros(len(label2id))
        for label_id in labels:
            label_tensor[label_id] = 1

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        labels_list.append(label_tensor)

    all_input_ids = t.stack(input_ids_list)
    all_attention_masks = t.stack(attention_masks_list)
    all_labels = t.stack(labels_list)

    data = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }

    t.save(data, cache_path)
    print(f"[✓] Saved {split} to cache.")

    return data


class PreTokenizedDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return {
        #     "input_ids": self.input_ids[idx],
        #     "attention_mask": self.attention_mask[idx],
        #     "labels": self.labels[idx]
        # }
        return self.input_ids[idx], self.labels[idx]


def get_dataloaders(tokenizer_name="bert-base-uncased", batch_size=32, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = preprocess_and_save("train", tokenizer, max_length)
    val_data   = preprocess_and_save("validation", tokenizer, max_length)
    test_data  = preprocess_and_save("test", tokenizer, max_length)

    train_dataset = PreTokenizedDataset(train_data)
    val_dataset   = PreTokenizedDataset(val_data)
    test_dataset  = PreTokenizedDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
