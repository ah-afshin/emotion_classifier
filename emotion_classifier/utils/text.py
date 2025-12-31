from transformers import AutoTokenizer


EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

label2id = {label: idx for idx, label in enumerate(EMOTIONS)}
id2label = {idx: label for label, idx in label2id.items()}

def get_vocab_size(tokenizer_name: str = "bert-base-uncased") -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.vocab_size