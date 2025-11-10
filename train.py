import torch as t
from torch import nn
from tqdm import tqdm



def train_bilstm(model: nn.Module, dl: t.utils.data.DataLoader, lr: float, epochs: int, method, device, logger):
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
        avg_epoch_loss = total_epoch_loss / len(dl)       
        print(f'epoch {epoch+1} | loss: {avg_epoch_loss:.4f}')
        logger.info(f'epoch {epoch+1} | loss: {avg_epoch_loss:.4f}')


def train_transformer(model: nn.Module, dl: t.utils.data.DataLoader, lr: float, epochs: int, mode, device, logger, encoder_lr=None):
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
        logger.error(f'ValueError: undefined mode: {mode}')
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
        avg_epoch_loss = total_epoch_loss / len(dl)
        print(f'epoch {epoch+1} | loss: {avg_epoch_loss:.4f}')
        logger.info(f'epoch {epoch+1} | loss: {avg_epoch_loss:.4f}')


if __name__=="__main__":
    import yaml
    from helpers import setup_device, setup_logger, save_config, set_seed, setup_path
    from models.bilstm import EmotionClassifierBiLSTM
    from models.transformer import EmotionClassifierTransformer
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger(config['paths']['log_path'])
    logger.info("Configuration:\n" + yaml.dump(config, sort_keys=False))
    device = setup_device(config)
    save_config('config.yaml', config['paths']['output_dir'])
    setup_path(config)
    set_seed(config, device)

    B = config['data']['batch_size']
    MAXLEN = config['data']['preprocessing']['max_length']
    TOKENIZER = config['data']['preprocessing']['tokenizer']
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=B, tokenizer_name=TOKENIZER, max_length=MAXLEN)
    
    LR = config['training']['lr']
    EPOCHS = config['training']['epochs']
    FINETUNE_LR = config['training']['finetune_lr']
    OPTIMIZER = config['training']['optimizer']

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
            train_bilstm(model, train_dl, LR, EPOCHS, config['model']['variant'], device, logger)
        
        case 'transformer':
            model = EmotionClassifierTransformer(
                            mode=config['model']['variant'],
                            model_name=config['model']['transformer']['transformer_model'],
                            dropout=config['model']['transformer']['dropout']
                    )
            print('training started.')
            logger.info('training started.')
            train_transformer(model, train_dl, LR, EPOCHS, 'fine-tune', device, logger, FINETUNE_LR)
            model.encoder.config.save_pretrained(config['paths']['encoder_checkpoint'])
        
        case _:
            logger.error(f"ValueError: Undefined model {config['model']['name']}.")
            raise ValueError(f"Undefined model {config['model']['name']}.")

    t.save(model.state_dict(), config['paths']['checkpoint'])
