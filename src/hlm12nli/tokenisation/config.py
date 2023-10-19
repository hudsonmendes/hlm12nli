# Python Built-in Modules
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hlm12NliTextTokeniserConfig:
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

    max_seq_len: int = field(default=1024)
    seq_len: int = field(default=None)
    special_tokens: set = field(default_factory=set)
    vocab: dict = field(default_factory=dict)
    token_pad: str = field(default="[PAD]")
    token_oov: str = field(default="[OOV]")
    token_str: str = field(default="[STR]")
    token_end: str = field(default="[END]")
    expr_subword: str = field(default="##")
    do_lowercase: bool = field(default=True)

    def __post_init__(self):
        self.special_tokens.add(self.token_pad)
        self.special_tokens.add(self.token_oov)
        self.special_tokens.add(self.token_str)
        self.special_tokens.add(self.token_end)

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int
                The size of the vocabulary.
        """
        return len(self.vocab)

    @property
    def vocab_by_token(self) -> dict:
        """
        Returns:
            dict
                The vocabulary as a dictionary.
        """
        return self.vocab if isinstance(self.vocab, dict) else {token: i for i, token in enumerate(self.vocab)}

    @property
    def vocab_by_token_id(self) -> dict:
        """
        Returns:
            dict
                The vocabulary as a dictionary.
        """
        v = self.vocab_by_token
        return {i: token for token, i in v.items()}
