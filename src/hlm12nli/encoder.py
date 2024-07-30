# Third-Party Libraries
import torch

# Local Folders
from .config import Hlm12NliConfig


class Hlm12NliEncoder(torch.nn.Module):
    def __init__(self, config: Hlm12NliConfig):
        super(Hlm12NliEncoder, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=len(config.vocab), embedding_dim=config.dim_embed, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=config.dim_embed, hidden_size=config.dim_lstm, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(in_features=config.dim_lstm, out_features=config.dim_out)

    def forward(self, x: torch.IntTensor) -> torch.FloatTensor:
        y = self.embeddings(x)
        _, (h, _) = self.lstm(y)
        y = h.squeeze(0)
        y = self.linear(y)
        return y
