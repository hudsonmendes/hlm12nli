# Python Built-in Modules
from dataclasses import dataclass


@dataclass(frozen=True)
class Hlm12NliEncoderConfig:
    vocab_size: int
    token_vec_dims: int
    token_id_pad: int
    hidden_dims: int
    hidden_bidir: bool
    attn_heads: int
    attn_dropout: float
    output_dims: int
