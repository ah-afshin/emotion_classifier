import pathlib, numpy, random, os, logging, shutil, torch as t


def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)

def save_config(config_path, output_dir):
    path = pathlib.Path(output_dir) / "config.yaml"
    shutil.copy(config_path, path)

def setup_device(config: dict):
    if config['general']['device'] == 'auto':
        # agnostic device setup
        return 'cuda' if t.cuda.is_available() else 'cpu'
    else:
        return config['general']['device']

def setup_path(path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

def set_seed(config: dict, device):
    SEED = config['general']['seed']
    t.manual_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if device == 'cuda':
        t.cuda.manual_seed_all(SEED)