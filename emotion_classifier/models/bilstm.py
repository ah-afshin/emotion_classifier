from typing import Literal
import torch as t
from torch import nn


class EmotionClassifierBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=3, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.lstm = nn.LSTM(
            input_size=128,             # same as embedding_dim
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True            # this way it's compatible with dataloaders
        )
        self.fc = nn.Linear(256*2, 28)  # hidden_size*2 -> output_size (=emotion_num)
    
    def forward(self,
                input_ids: t.Tensor,
                attention_mask: t.Tensor,
                method: Literal['max-pool', 'last-token']
        ) -> t.Tensor:
                                            # shape: [B, seq_len]
        embed = self.embedding(input_ids)   # embedding vectors, shape: [B, seq_len, hidden*2]
                                            # shape: [B, seq_len, embedding_dim]
        out, _ = self.lstm(embed)           # all outputs for all time steps (all inputs of a sequence)
                                            # shape: [B, seq_len, hidden_size*2]

        match method:
            case 'max-pool':
                # In this method we choose the largest output instead of last one. 
                # Before pooling, set the outputs of padding tokens to a very small number
                # This ensures they are not selected by the max-pooling operation.
                mask = attention_mask.unsqueeze(-1).expand(out.shape).float()
                out = out * mask
                out[mask == 0] = -1e9
                pooled = t.max(out, dim=1).values   # max-pooling over the time dimension
                                                    # shape: [B, hidden_size*2]
                return self.fc(pooled)
            
            case 'last-token':
                # In this method we use the last meaningful token.
                # First we use attention_mask to find the actual len of seq
                # Sum of attention_mask tells us how many non-padding tokens are there. 
                sequence_lengths = attention_mask.sum(dim=1) - 1    # -1 'cuz scale starts from 0
                                                                    # shape: [B]
                batch_size = out.shape[0]                           # Use this len to find last meaningful token
                batch_indices = t.arange(0, batch_size).to(out.device)
                last_outputs = out[batch_indices, sequence_lengths, :]
                                                                    # shape: [B, hidden_size*2]
                return self.fc(last_outputs)
            
            case _:
                raise ValueError(f'undefined method {method}')
        # last_token = out[:, -1, :]          # :→all_Batch , -1→last_in_seq , :→all_hidden_dims
        # return self.fc(last_token)          # we only use the final output to classify the sentence
