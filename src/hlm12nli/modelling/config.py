# Python Built-in Modules
from dataclasses import dataclass, field

# Third-Party Libraries
from nest_ml.text import NestMLTextEncoderConfigBase


@dataclass(frozen=True)
class Hlm12NliConfig(NestMLTextEncoderConfigBase):
    """
    Configuration for the HLM12NLI model.

    Attributes:
        model_name: str
            Name of the model, inherited from `NestMLModelConfigBase`.
        hidden_state_dims: int
            The number of hidden state dimensions, inherited from `NestMLEncoderConfigBase`
        vocab_size: int
            The total number of tokens in the vocabulary of the tokeniser, inherited from `NestMLTextEncoderConfigBase`
        token_vec_dims: int
            The number of dimensions of the trainable token vector embeddings.
        token_id_pad: int
            The id of the padding token, used by the tokeniser.
        attn_heads: int
            The number of attention heads in the multi-head attention layer.
        attn_dropout: float
            The dropout rate of the multi-head attention layer.
        output_dims: int
            The number of output dimensions, inherited from `NestMLEncoderConfigBase`
    """

    token_vec_dims: int = field()
    token_id_pad: int = field()
    attn_heads: int = field()
    attn_dropout: float = field()
