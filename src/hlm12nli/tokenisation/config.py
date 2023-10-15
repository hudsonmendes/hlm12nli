# Python Built-in Modules
from dataclasses import dataclass, field

# Third-Party Libraries
from hlm12ml.text import Hlm12TextTokeniserConfigBase


@dataclass(frozen=True)
class Hlm12NliTextTokeniserConfig(Hlm12TextTokeniserConfigBase):
    """
    Configuration for the HLM12 NLI text tokeniser.
    Inherits from the base `Hlm12TextTokeniserConfigBase` and all its fields.

    Attributes:
        max_seq_len: int
            The maximum sequence length.
        seq_len: int | None
            The sequence length to pad to. If None, the maximum sequence length is used.
        special_tokens: Set[str]
            The set of special tokens. The padding and out-of-vocabulary tokens are always added.
        vocab: List[str] | Dict[str, int]
            The vocabulary mapping tokens to indices.
        token_pad: str
            The token used for padding.
        token_oov: str
            The token used for out-of-vocabulary tokens.
        token_str: str
            The token used for the start of a sentence.
        token_end: str
            The token used for the end of a sentence.
        expr_subword: str
            The expression used to mark subwords.
        do_lowercase: bool
            Whether to lowercase the input.
    """

    token_str: str = field(default="[STR]")
    token_end: str = field(default="[END]")
    expr_subword: str = field(default="##")
    do_lowercase: bool = field(default=True)
