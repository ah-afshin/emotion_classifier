from datetime import datetime

import yaml
import torch as t
from torch import nn

from emotion_classifier.models import build_model
from emotion_classifier.data import get_dataloaders
from emotion_classifier.utils import setup_device, setup_path, save_config, set_seed, setup_logger
from .trainer import train_bilstm, train_transformer



def run_training(config, config_path, output_dir):
    LR = config['training']['lr']
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    threshold = config['general']['threshold']
    finetune_LR = config['training'].get('finetune_lr', None)
    variant = config['model']['variant']
    # optimizer = config['training']['optimizer']

    path =  output_dir / f"{config['model']['name']}-{config['model']['variant']}" / datetime.now().strftime('%Y-%m-%d_%H-%M')
    device = setup_device(config)

    def save_model(state_dict):
        t.save(state_dict, path/'best_model.pt')
    
    setup_path(path)
    save_config(config_path, path)
    set_seed(config, device)
    logger = setup_logger(path/'train.log')
    logger.info("Configuration:\n" + yaml.dump(config, sort_keys=False))
    print(f'using device: {device}')

    B = config['data']['batch_size']
    max_len = config['data']['preprocessing']['max_length']
    tokenizer = config['data']['preprocessing']['tokenizer']
    train_dl, val_dl, _ = get_dataloaders(batch_size=B, tokenizer_name=tokenizer, max_length=max_len)
    
    model = build_model(config)
    match config['model']['name']:
        case 'bilstm':
            train_bilstm(model, train_dl, val_dl, epochs, LR, threshold, patience, save_model, logger, device)
        case 'transformer':
            train_transformer(model, train_dl, val_dl, epochs, LR, finetune_LR, variant, threshold, patience, save_model, logger, device)
            model.encoder.config.save_pretrained(path/'bert/')
        case _:
            logger.error(f"ValueError: Undefined model {config['model']['name']}.")
            raise ValueError(f"Undefined model {config['model']['name']}.")
    t.save(model.state_dict(), path/'last_epoch.pt')
