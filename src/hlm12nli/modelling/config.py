# Python Built-in Modules
from dataclasses import dataclass

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
        output_dims: int
            The number of output dimensions, inherited from `NestMLEncoderConfigBase`
        vocab_size: int
            The total number of tokens in the vocabulary of the tokeniser, inherited from `NestMLTextEncoderConfigBase`
    """

    pass
