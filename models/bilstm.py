import torch as t
from torch import nn


class EmotionClassifierBiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            dropout=0.25,
            batch_first=True            # this way it's compatible with dataloaders
        )
        self.fc = nn.Linear(256*2, 28)  # hidden_size*2 -> output_size (=emotion_num)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        embed = self.embedding(x)       # embedding vectors, shape: [B, seq_len, hidden*2]
        out, _ = self.lstm(embed)       # all outputs for all time steps (all inputs of a sequence)
        last_token = out[:, -1, :]      # :→all_Batch , -1→last_in_seq , :→all_hidden_dims
        return self.fc(last_token)      # we only use the final output to classify the sentence
