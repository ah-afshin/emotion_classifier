import yaml
import torch as t
import numpy as np
from torch import nn
from tqdm import tqdm
from utils.metrics import compute_metrics



def validation(model: nn.Module, validation_dl: t.utils.data.DataLoader, criterion, threshold, device, method=None):
    total_loss = 0
    y_true_list = []
    y_pred_list = []
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

            prob = t.sigmoid(logit)
            pred = (prob > threshold).int().cpu().numpy()
            y_true_list.append(y.cpu().numpy().astype(int))
            y_pred_list.append(pred)
        
        y_true = np.concatenate(y_true_list, axis=0)        # shape: (n_samples, num_labels)
        y_pred = np.concatenate(y_true_list, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics['val-loss'] = total_loss / len(validation_dl)
    return metrics  


def train_bilstm(model: nn.Module, train_dl: t.utils.data.DataLoader, val_dl: t.utils.data.DataLoader, config, logger):
    LR = config['training']['lr']
    epochs = config['training']['epochs']
    device = config['device']
    method = config['model']['variant']
    path = config['path']
    patience = config['training']['patience']
    threshold = config['general']['threshold']
    # optimizer = config['training']['optimizer']
    
    optim = t.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_metric = float('-inf')
    patience_counter = 0

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
        metrics = validation(model, val_dl, criterion, threshold, device, method)
        print(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {metrics["val-loss"]:.4f} | f1 score: {metrics["f1-micro"]}')
        logger.info(f'\nepoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {metrics["val-loss"]:.4f}')
        logger.info("Metrics:\n" + yaml.dump(metrics, sort_keys=False))

        # if val_loss < best_loss:
        if metrics['f1-micro']>best_metric:
            best_metric = metrics['f1-micro']
            patience_counter = 0
            t.save(model.state_dict(), path+'best_model.pt')
        else:
            patience_counter += 1
            logger.warning('Metrics on Validation dataset did not improved during this epoch of training.')
            if patience_counter>=patience:
                print('Trainig has stopped. (early stopping triggered)')
                logger.info('Early stopping triggered.')
                break
    

def train_transformer(model: nn.Module, train_dl: t.utils.data.DataLoader, val_dl: t.utils.data.DataLoader, config, logger):
    LR = config['training']['lr']
    epochs = config['training']['epochs']
    finetune_LR = config['training'].get('finetune_lr', None)
    device = config['device']
    mode = config['model']['variant']
    path = config['path']
    patience = config['training']['patience']
    threshold = config['general']['threshold']
    # optimizer = config['training']['optimizer']

    best_metric = float('-inf')
    patience_counter = 0
    model.to(device=device)

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
        model.train()
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
        metrics = validation(model, val_dl, criterion, threshold, device)
        print(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {metrics["val-loss"]:.4f} | f1 score: {metrics["f1-micro"]}')
        logger.info(f'epoch {epoch+1} | train_loss: {avg_epoch_loss:.4f} | validation_loss: {metrics["val-loss"]:.4f}')

        # if val_loss < best_loss:
        if metrics['f1-micro'] > best_metric:
            best_metric = metrics['f1-micro']
            patience_counter = 0
            t.save(model.state_dict(), path+'best_model.pt')
        else:
            patience_counter += 1
            logger.warning('Metrics on Validation dataset did not improved during this epoch of training.')
            if patience_counter>=patience:
                print('Trainig has stopped. (early stopping triggered)')
                logger.info('Early stopping triggered.')
                break


if __name__=="__main__":
    from helpers import setup_device, setup_logger, save_config, set_seed, setup_path
    from models.bilstm import EmotionClassifierBiLSTM
    from models.transformer import EmotionClassifierTransformer
    from data.preprocess import get_dataloaders
    from utils.text import get_vocab_size
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    path = config['output_dir'] + f"{config['model']['name']}-{config['model']['variant']}/"
    device = setup_device(config)
    print(f'using device: {device}')
    config['device'] = device
    config['path'] = path
    
    setup_path(path)
    save_config('config.yaml', path)
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
