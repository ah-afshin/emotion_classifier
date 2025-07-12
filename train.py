import torch as t
from torch import nn



def train_bilstm(model:nn.Module, dl: t.utils.data.DataLoader, lr: float, epochs: int, device):
    optim = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            total_epoch_loss += loss.item()
        print(f'epoch {epoch+1} | loss: {total_epoch_loss}')


if __name__=="__main__":
    from models.bilstm import EmotionClassifierBiLSTM
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    from config import B, LR, EPOCHS

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=B)
    model = EmotionClassifierBiLSTM(vocab_size=get_vocab_size()).to(device=device)
    print('training started.')
    train_bilstm(model, train_dl, LR, EPOCHS, device)
    t.save(model.state_dict(), "checkpoints/bilstm.pt")
