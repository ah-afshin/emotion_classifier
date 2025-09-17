import torch as t
from torch import nn
from tqdm import tqdm



def train_bilstm(model: nn.Module, dl: t.utils.data.DataLoader, lr: float, epochs: int, method, device):
    optim = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch in progress_bar:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            logit = model(x, mask, method)
            loss = criterion(logit, y.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            total_epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())        
        print(f'epoch {epoch+1} | loss: {total_epoch_loss:.4f}')


def train_transformer(model: nn.Module, dl: t.utils.data.DataLoader, lr: float, epochs: int, mode, device, encoder_lr=None):
    model.to(device=device)
    model.train()

    if mode=='feature-extract':
        optim = t.optim.Adam(model.head.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
    elif mode=='fine-tune':
        optimizer_grouped_parameters = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": lr}
        ]
        optim = t.optim.AdamW(optimizer_grouped_parameters)     # AdamW works better with transformers
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'undefined mode: {mode}')

    for epoch in range(epochs):
        total_epoch_loss = 0
        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch in progress_bar:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            logit = model(x, mask)
            loss = criterion(logit, y.float())
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())        
        print(f'epoch {epoch+1} | loss: {total_epoch_loss:.4f}')


if __name__=="__main__":
    from sys import argv, exit

    from models.bilstm import EmotionClassifierBiLSTM
    from models.transformer import EmotionClassifierTransformer
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    from config import B, LR, EPOCHS, FINETUNE_LR

    device = 'cuda' if t.cuda.is_available() else 'cpu'

    if len(argv) != 3:
        print('Usage: python train.py <model> <method>')
        exit(1)

    match (argv[1], argv[2]):
        case ('bilstm', 'max-pool'):
            train_dl, val_dl, test_dl = get_dataloaders(batch_size=B)
            model = EmotionClassifierBiLSTM(vocab_size=get_vocab_size())
            print('training started.')
            train_bilstm(model, train_dl, LR, EPOCHS, 'max-pool', device)
            t.save(model.state_dict(), "checkpoints/bilstm_maxpool.pt")
        
        case ('bilstm', 'last-token'):
            train_dl, val_dl, test_dl = get_dataloaders(batch_size=B)
            model = EmotionClassifierBiLSTM(vocab_size=get_vocab_size())
            print('training started.')
            train_bilstm(model, train_dl, LR, EPOCHS, 'last-token', device)
            t.save(model.state_dict(), "checkpoints/bilstm_lasttoken.pt")
        
        case ('transformer', 'feature-extract'):
            train_dl, val_dl, test_dl = get_dataloaders(batch_size=B, tokenizer_name='distilbert-base-uncased')
            model = EmotionClassifierTransformer('feature-extract')
            print('training started.')
            train_transformer(model, train_dl, LR, EPOCHS, 'feature-extract', device)
            model.encoder.config.save_pretrained("checkpoints/bert")
            t.save(model.state_dict(), "checkpoints/transformer_featureextract.pt")
        
        case ('transformer', 'fine-tune'):
            train_dl, val_dl, test_dl = get_dataloaders(batch_size=B, tokenizer_name='distilbert-base-uncased')
            model = EmotionClassifierTransformer('fine-tune')
            print('training started.')
            train_transformer(model, train_dl, LR, EPOCHS, 'fine-tune', device, FINETUNE_LR)
            model.encoder.config.save_pretrained("checkpoints/bert")
            t.save(model.state_dict(), "checkpoints/transformer_finetune.pt")
        
        case _:
            print('Undefined model and/or method.')
