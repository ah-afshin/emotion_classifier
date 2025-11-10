import pathlib, numpy, random, os, logging, shutil, torch as t


def save_config(config_path, output_dir):
    path = pathlib.Path(output_dir) / "config.yaml"
    shutil.copy(config_path, path)

def setup_device(config: dict):
    if config['general']['device'] == 'auto':
        # agnostic device setup
        return 'cuda' if t.cuda.is_available() else 'cpu'
    else:
        return config['general']['device']

def setup_path(config: dict):
    os.makedirs(
        os.path.dirname(config['paths']['checkpoint']),
        exist_ok=True
    )

def set_seed(config: dict, device):
    SEED = config['general']['seed']
    t.manual_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if device == 'cuda':
        t.cuda.manual_seed_all(SEED)