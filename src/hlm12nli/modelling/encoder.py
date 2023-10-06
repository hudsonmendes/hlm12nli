# Third-Party Libraries
import torch
import transformers

# Local Folders
from .config import Hlm12NliConfig


class Hlm12NliEncoder(transformers.PreTrainedModel):
    def __init__(self, config: Hlm12NliConfig):
        self.embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.lstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_dims,
            bidirectional=config.lstm_bidir,
            batch_first=True,
        )
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=config.seq_len,
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.embeddings(input_ids)
        y, _ = self.lstm(y)
        attn, _ = self.mha(y, y, y, key_padding_mask=attention_mask)
        y = y + attn
        return y
