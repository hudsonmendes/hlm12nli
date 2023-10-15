# Python Built-in Modules
from typing import List

# Third-Party Libraries
from hlm12ml.text import Hlm12TextTokeniserBase

# Local Folders
from .config import Hlm12NliTextTokeniserConfig
from .joining import Hlm12NliTokeniserJoiner
from .output import Hlm12NliTokeniserOutput
from .splitter import Hlm12NliTokeniserSplitter


class Hlm12NliTokeniser(Hlm12TextTokeniserBase[Hlm12NliTextTokeniserConfig, Hlm12NliTokeniserOutput]):
    """
    Tokeniser for the Hlm12Nli model, implemented using the wordpiece algorithm
    with padding, truncating and special tokens capabilities.
    Inherits from `Hlm12TextTokeniserBase`.

    Attributes:
        max_seq_len: int
            The maximum sequence length.
        seq_len: int | None
            The sequence length to pad to. If None, the maximum sequence length is used.
        special_tokens: Set[str]
            The set of special tokens.s
        vocab: Dict[str, int]
            The vocabulary mapping tokens to indices.
        token_pad: str
            The token used for padding.
        token_oov: str
            The token used for out-of-vocabulary tokens.
        token_str: str
            The token used to envelop the input, marking the start of each sentence.
        token_end: str
            The token used to mark the end of each sentence.
    """

    token_str: str
    token_end: str
    do_lowercase: bool

    def __init__(self, config: Hlm12NliTextTokeniserConfig):
        """
        Constructs a new Hlm12NliTokeniser.

        Args:
            config: Hlm12NliTextTokeniserConfig
                The tokeniser configuration.
        """
        super().__init__(config=config)
        self.token_str = config.token_str
        self.token_end = config.token_end
        self.do_lowercase = config.do_lowercase
        self._splitter_fn = Hlm12NliTokeniserSplitter(
            vocab=config.vocab,
            token_oov=self.token_oov,
            expr_subword=config.expr_subword,
        )
        self._joiner_fn = Hlm12NliTokeniserJoiner(
            special_tokens=config.special_tokens,
            expr_subword=config.expr_subword,
        )

    def perform_splitting(self, x: List[str]) -> List[List[str]]:
        if self.do_lowercase:
            x = [xi.lower() for xi in x]
        y = self._splitter_fn(x)
        y = [[self.token_str] + yi + [self.token_end] for yi in y]
        return y

    def perform_joining(self, x: List[List[str]]) -> List[str]:
        return self._joiner_fn(x)
