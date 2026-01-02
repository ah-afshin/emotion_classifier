import torch as t

def setup_device(config: dict):
    if config['general']['device'] == 'auto':
        # agnostic device setup
        return 'cuda' if t.cuda.is_available() else 'cpu'
    else:
        return config['general']['device']
