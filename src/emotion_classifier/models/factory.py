import torch as t

from emotion_classifier.utils import get_vocab_size
from .bilstm import EmotionClassifierBiLSTM
from .transformer import EmotionClassifierTransformer



def build_model(config, trained_model=None):
    match config['model']['name']:
        case 'bilstm':
            model = EmotionClassifierBiLSTM(
                            variant=config['model']['variant'],
                            vocab_size=get_vocab_size(),
                            hidden_size=config['model']['bilstm']['hidden_size'],
                            num_layers=config['model']['bilstm']['num_layers'],
                            dropout=config['model']['bilstm']['dropout']
                    )
        case 'transformer':
            model = EmotionClassifierTransformer(
                            mode=config['model']['variant'],
                            model_name=config['model']['transformer']['transformer_model'],
                            dropout=config['model']['transformer']['dropout']
                    )
        case _:
            raise ValueError('Unknown model.')
        
    if trained_model:
        model.load_state_dict(t.load(trained_model/'best_model.pt', weights_only=True))
    
    return model
