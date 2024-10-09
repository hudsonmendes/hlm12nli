# Python Built-in Modules
from dataclasses import dataclass
from typing import List

# Third-Party Libraries
import torch

# Local Folders
from .tokenisation import Hlm12NliTextTokenisation


@dataclass(frozen=True)
class Hlm12NliConfig:
    n_vocab: int
    dim_embed: int
    dim_lstm: int
    dim_out: int


class Hlm12NliEncoder(torch.nn.Module):
    def __init__(self, config: Hlm12NliConfig):
        super(Hlm12NliEncoder, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=config.dim_embed, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=config.dim_embed, hidden_size=config.dim_lstm, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(in_features=config.dim_lstm, out_features=config.dim_out)

    def forward(self, x: Hlm12NliTextTokenisation) -> torch.FloatTensor:
        y = self.embeddings(x.ids)
        _, (h, _) = self.lstm(y)
        y = h.squeeze(0)
        y = self.linear(y)
        return y
