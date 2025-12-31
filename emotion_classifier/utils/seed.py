import numpy, random, torch as t


def set_seed(config: dict, device):
    SEED = config['general']['seed']
    t.manual_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if device == 'cuda':
        t.cuda.manual_seed_all(SEED)
