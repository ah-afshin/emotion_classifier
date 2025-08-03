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
            # x, y = batch[0].to(device), batch[1].to(device)
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


if __name__=="__main__":
    from models.bilstm import EmotionClassifierBiLSTM
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    from config import B, LR, EPOCHS

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=B)
    
    model = EmotionClassifierBiLSTM(vocab_size=get_vocab_size())
    print('training started.')
    train_bilstm(model, train_dl, LR, EPOCHS, 'max-pool', device)
    t.save(model.state_dict(), "checkpoints/bilstm_maxpool.pt")
