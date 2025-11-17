import torch as t
from torch import nn
from tqdm import tqdm



def validation(model: nn.Module, validation_dl: t.utils.data.DataLoader, criterion, device, method=None):
    total_loss = 0
    model.eval()
    with t.no_grad():
        for batch in validation_dl:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            if method:
                logit = model(x, mask, method)
            else:
                logit = model(x, mask)
            loss = criterion(logit, y.float())
            total_loss += loss.item()
    return total_loss / len(validation_dl)  


def train_bilstm(model: nn.Module, train_dl: t.utils.data.DataLoader, val_dl: t.utils.data.DataLoader, config, logger):
    LR = config['training']['lr']
    epochs = config['training']['epochs']
    device = config['device']
    method = config['model']['variant']
    path = config['path']
    # optimizer = config['training']['optimizer']
    
    optim = t.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

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
        avg_epoch_loss = total_epoch_loss / len(train_dl)
        val_loss = validation(model, val_dl, criterion, device, method)
        print(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {val_loss:.4f}')
        logger.info(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            t.save(model.state_dict(), path+'best_model.pt')
        else:
            logger.warning('Metrics on Validation dataset did not improved during this epoch of training.')

def train_transformer(model: nn.Module, train_dl: t.utils.data.DataLoader, val_dl: t.utils.data.DataLoader, config, logger):
    LR = config['training']['lr']
    epochs = config['training']['epochs']
    finetune_LR = config['training'].get('finetune_lr', None)
    device = config['device']
    mode = config['model']['variant']
    path = config['path']
    # optimizer = config['training']['optimizer']

    best_loss = float('inf')
    model.to(device=device)
    model.train()

    if mode=='feature-extract':
        optim = t.optim.Adam(model.head.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()
    elif mode=='fine-tune':
        optimizer_grouped_parameters = [
            {"params": model.encoder.parameters(), "lr": finetune_LR},
            {"params": model.head.parameters(), "lr": LR}
        ]
        optim = t.optim.AdamW(optimizer_grouped_parameters)     # AdamW works better with transformers
        criterion = nn.BCEWithLogitsLoss()
    else:
        logger.error(f'ValueError: undefined mode: {mode}')
        raise ValueError(f'undefined mode: {mode}')

    for epoch in range(epochs):
        total_epoch_loss = 0
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

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
        avg_epoch_loss = total_epoch_loss / len(train_dl)
        val_loss = validation(model, val_dl, criterion, device)
        print(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {val_loss:.4f}')
        logger.info(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            t.save(model.state_dict(), path+'best_model.pt')
        else:
            logger.warning('Metrics on Validation dataset did not improved during this epoch of training.')


if __name__=="__main__":
    import yaml
    from helpers import setup_device, setup_logger, save_config, set_seed, setup_path
    from models.bilstm import EmotionClassifierBiLSTM
    from models.transformer import EmotionClassifierTransformer
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    path = config['output_dir'] + f"{config['model']['name']}-{config['model']['variant']}/"
    device = setup_device(config)
    config['device'] = device
    config['path'] = path
    
    save_config('config.yaml', path)
    setup_path(path)
    set_seed(config, device)
    logger = setup_logger(path+'train.log')
    logger.info("Configuration:\n" + yaml.dump(config, sort_keys=False))

    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    tokenizer = config['data']['preprocessing']['tokenizer']
    train_dl, val_dl, _ = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)

    match config['model']['name']:
        case 'bilstm':
            model = EmotionClassifierBiLSTM(
                            vocab_size=get_vocab_size(),
                            hidden_size=config['model']['bilstm']['hidden_size'],
                            num_layers=config['model']['bilstm']['num_layers'],
                            dropout=config['model']['bilstm']['dropout']
                    )
            print('training started.')
            logger.info('training started.')
            train_bilstm(model, train_dl, val_dl, config, logger)
        
        case 'transformer':
            model = EmotionClassifierTransformer(
                            mode=config['model']['variant'],
                            model_name=config['model']['transformer']['transformer_model'],
                            dropout=config['model']['transformer']['dropout']
                    )
            print('training started.')
            logger.info('training started.')
            train_transformer(model, train_dl, val_dl, config, logger)
            model.encoder.config.save_pretrained(path+'bert/')
        
        case _:
            logger.error(f"ValueError: Undefined model {config['model']['name']}.")
            raise ValueError(f"Undefined model {config['model']['name']}.")

    t.save(model.state_dict(), path+'last_epoch.pt')
