from typing import Literal
import torch as t
from torch import nn
from transformers import AutoModel


class EmotionClassifierTransformer(nn.Module):
    def __init__(self,
                mode: Literal ['feature-extract', 'fine-tune'],
                model_name="distilbert-base-uncased",
                num_labels=28,
                dropout=0.3
        ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        match mode:
            case 'feature-extract':
                # Feature Extraction means we only train a head to work with embedding vecs
                # So we need to freeze our transformer model.
                for param in self.encoder.parameters():
                    param.requires_grad = False
            case 'fine-tune':
                # Fine Tuning is when we train the encoder transformer too (usually with a lower LR)
                # It's more accurate, But also slower. We enable gradients here.
                for param in self.encoder.parameters():
                    param.requires_grad = True
            case _:
                raise ValueError(f'undefined mode {mode}')
        
        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),        # these two layers are added just in case we wanted to make the head deeper
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, num_labels),
            # nn.Sigmoid()    # multi-label classification (commented so we use BCEWithLogits criterion)
        )

    def forward(self,
                input_ids: t.Tensor,
                attention_mask: t.Tensor
        ) -> t.Tensor:
        embedding_vec = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = embedding_vec.last_hidden_state[:, 0, :]   # CLS token representation
        return self.head(pooled)                            # this is the final classification result.
    