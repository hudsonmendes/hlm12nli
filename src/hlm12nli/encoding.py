# Python Built-in Modules
from typing import List

# Third-Party Libraries
import torch

# Local Folders
from .hyperparams import Hlm12NliHyperparams
from .tokenisation import Hlm12NliTextTokenisation


class Hlm12NliEncoder(torch.nn.Module):
    def __init__(self, hyperparams: Hlm12NliHyperparams):
        super(Hlm12NliEncoder, self).__init__()
        vocab_len = len(hyperparams.tokeniser.vocab)
        idx_pad = hyperparams.tokeniser.vocab.index("<pad>")
        embed_dim = hyperparams.encoder.embed_dim
        lstm_dim = hyperparams.encoder.lstm_dim
        lstm_bidirectional = hyperparams.encoder.lstm_bidirectional
        out_dim = hyperparams.encoder.out_dim
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim, padding_idx=idx_pad)
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=lstm_dim, batch_first=True, bidirectional=lstm_bidirectional)
        self.linear = torch.nn.Linear(in_features=lstm_dim, out_features=out_dim)

    def forward(self, x: Hlm12NliTextTokenisation) -> torch.FloatTensor:
        y = self.embeddings(x.ids)
        _, (h, _) = self.lstm(y)
        y = h.squeeze(0)
        y = self.linear(y)
        return y
