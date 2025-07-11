from data.preprocess import get_dataloaders
_ = get_dataloaders()
from datasets import config
print(config.HF_DATASETS_CACHE) # C:\Users\PC\.cache\huggingface\datasets
