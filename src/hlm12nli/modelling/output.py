# Python Built-in Modules
from dataclasses import dataclass

# Third-Party Libraries
from nest_ml.text import NestMLTextEncoderOutputBase


@dataclass(frozen=True)
class Hlm12NliOutput(NestMLTextEncoderOutputBase):
    """
    Output for the HLM12NLI model.

    Attributes:
        embeddings: List[List[float]]
            Bath ofsentence representations.
        last_hidden_state: List[List[List[float]]]
            Batch of hidden states for each token in each sentence.
    """

    pass
