import os
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch as t
from torch.nn.utils.rnn import pad_sequence


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
    # input_ids_list = []
    # attention_masks_list = []
    # labels_list = []
    processed_data = []

    for item in dataset:
        text = item["text"]
        labels = item["labels"]

        # Tokenize without padding
        encoding = tokenizer(text, truncation=True, max_length=max_length) # max_length is just a safeguard

        label_tensor = t.zeros(len(label2id))
        for label_id in labels:
            label_tensor[label_id] = 1

        # input_ids_list.append(input_ids)
        # attention_masks_list.append(attention_mask)
        # labels_list.append(label_tensor)
        processed_data.append({
            "input_ids": t.tensor(encoding["input_ids"]),
            "attention_mask": t.tensor(encoding["attention_mask"]),
            "labels": label_tensor
        })

    # all_input_ids = t.stack(input_ids_list)
    # all_attention_masks = t.stack(attention_masks_list)
    # all_labels = t.stack(labels_list)

    # data = {
    #     "input_ids": all_input_ids,
    #     "attention_mask": all_attention_masks,
    #     "labels": all_labels
    # }

    # t.save(data, cache_path)
    t.save(processed_data, cache_path)
    print(f"[✓] Saved {split} to cache.")
    # return data
    return processed_data


class GoEmotionsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.input_ids = data["input_ids"]
        # self.attention_mask = data["attention_mask"]
        # self.labels = data["labels"]

    def __len__(self):
        return len(self.data)
        # return len(self.input_ids)

    def __getitem__(self, idx):
        # return self.input_ids[idx], self.labels[idx]
        return self.data[idx]

class PadCollator:
    """This class, acts as a function that  will handle dynamic padding for each batch"""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad sequences to the max length in this batch
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
        # Stack labels
        labels_stacked = t.stack(labels)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "labels": labels_stacked
        }

# def create_collate_fn(pad_token_id):
    
#     def collate_fn(batch):
#         input_ids = [item['input_ids'] for item in batch]
#         attention_masks = [item['attention_mask'] for item in batch]
#         labels = [item['labels'] for item in batch]

#         # Pad sequences to the max length in this batch
#         input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
#         attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
#         # Stack labels
#         labels_stacked = t.stack(labels)

#         return {
#             "input_ids": input_ids_padded,
#             "attention_mask": attention_masks_padded,
#             "labels": labels_stacked
#         }
#     return collate_fn


def get_dataloaders(tokenizer_name="bert-base-uncased", batch_size=32, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = preprocess_and_save("train", tokenizer, max_length)
    val_data   = preprocess_and_save("validation", tokenizer, max_length)
    test_data  = preprocess_and_save("test", tokenizer, max_length)

    train_dataset = GoEmotionsDataset(train_data)
    val_dataset   = GoEmotionsDataset(val_data)
    test_dataset  = GoEmotionsDataset(test_data)

    # collate_func = create_collate_fn(tokenizer.pad_token_id)
    collate_func = PadCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_func, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_func, num_workers=4)

    return train_loader, val_loader, test_loader
