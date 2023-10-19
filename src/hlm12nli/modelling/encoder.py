# Third-Party Libraries
import torch

# Local Folders
from .config import Hlm12NliConfig


class Hlm12NliEncoder(torch.nn.Module):
    def __init__(self, config: Hlm12NliConfig):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.token_vec_dims,
            padding_idx=config.token_id_pad,
        )
        self.lstm = torch.nn.LSTM(
            input_size=config.token_vec_dims,
            hidden_size=config.hidden_dims,
            bidirectional=config.hidden_bidir,
            batch_first=True,
        )
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_dims,
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.layer_norm = torch.nn.LayerNorm(
            config.hidden_dims,
        )
        self.linear = torch.nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.output_dims,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        y = self.embeddings(input_ids)
        y, _ = self.lstm(y)
        attn, _ = self.mha(y, y, y, key_padding_mask=input_mask)
        y = y + attn
        y = self.layer_norm(y)
        y = self.linear(y)
        return y
