# Python Built-in Modules
from typing import Dict, List

# Local Folders
from .exceptions import (
    Hlm12NliTokeniserBatchRequiredError,
    Hlm12NliTokeniserMissingSpecialTokenInVocabError,
    Hlm12NliTokeniserSeqLenError,
    Hlm12NliTokeniserVocabTokenCasingError,
)
from .joiner import Hlm12NliTokeniserJoiner
from .splitter import Hlm12NliTokeniserSplitter


class Hlm12NliTokeniser:
    """
    Tokeniser for the Hlm12Nli model, implemented using the wordpiece algorithm
    with padding, truncating and special tokens capabilities.
    """

    vocab: Dict[str, int]
    do_lowercase: bool
    special_tokens: List[str]
    seq_len: int | None
    max_seq_len: int
    expr_subtoken: str

    def __init__(
        self,
        vocab: Dict[str, int] | List[str],
        do_lowercase=True,
        seq_len: int | None = None,
        max_seq_len: int = 1024,
        expr_subtoken: str = "##",
        token_oov: int = "[OOV]",
        token_pad: int = "[PAD]",
        token_str: int = "[STR]",
        token_end: int = "[END]",
    ):
        """
        Constructs a new Hlm12NliTokeniser.

        Args:
            vocab: The vocabulary to use.
            do_lowercase: Whether to lowercase the tokens, default is True.
            token_oov: The out-of-vocabulary token, default is "[OOV]".
            token_pad: The padding token, default is "[PAD]".
            token_str: The start-of-sentence token, default is "[STR]".
            token_end: The end-of-sentence token, default is "[END]".
            seq_len: The fixed length of the sequence, None results in padding to max len in the batch, default is None.
            max_seq_len: The maximum sequence length, after which data will be truncated, default is 1024.
        """
        self.do_lowercase = do_lowercase
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.expr_subtoken = expr_subtoken
        self.token_str = token_str
        self.token_end = token_end
        self.token_pad = token_pad
        self.special_tokens = [token_oov, token_pad, token_str, token_end]
        self.vocab = vocab if isinstance(vocab, dict) else {token: i for i, token in enumerate(vocab)}
        self._ensure_seq_length_constraints(self.seq_len, self.max_seq_len)
        self._ensure_respects_casing(vocab=self.vocab, do_lowercase=self.do_lowercase, ignore=self.special_tokens)
        self._ensure_special_tokens_in_vocab(vocab=self.vocab, special_tokens=self.special_tokens)
        self._splitter_fn = Hlm12NliTokeniserSplitter(self.vocab, token_oov=token_oov, expr_subtoken=self.expr_subtoken)
        self._joiner_fn = Hlm12NliTokeniserJoiner(self.special_tokens, expr_subword=self.expr_subtoken)

    @staticmethod
    def _ensure_seq_length_constraints(seq_len: int, max_seq_len: int) -> None:
        if seq_len is not None and seq_len > max_seq_len:
            raise Hlm12NliTokeniserSeqLenError(seq_len=seq_len, max_seq_len=max_seq_len)

    @staticmethod
    def _ensure_respects_casing(vocab: Dict[str, int], do_lowercase: bool, ignore: List[str]) -> None:
        if do_lowercase:
            for token in vocab.keys():
                if token not in ignore and token != token.lower():
                    raise Hlm12NliTokeniserVocabTokenCasingError(token=token)

    @staticmethod
    def _ensure_special_tokens_in_vocab(vocab: Dict[str, int], special_tokens: List[str]) -> None:
        for token in special_tokens:
            if token not in vocab:
                raise Hlm12NliTokeniserMissingSpecialTokenInVocabError(token=token)

    @staticmethod
    def _ensure_batch(x: List[str]) -> None:
        if not isinstance(x, list):
            raise Hlm12NliTokeniserBatchRequiredError()

    def tokenize(self, x: List[str]) -> List[List[str]]:
        self._ensure_batch(x)
        if self.do_lowercase:
            x = [xi.lower() for xi in x]
        y = self._splitter_fn(x)
        y = [self._envelop(yi) for yi in y]
        y = self._pad(y)
        return y

    def _envelop(self, tokens: List[str]) -> List[str]:
        return [self.token_str] + tokens + [self.token_end]

    def _pad(self, batch: List[List[str]]) -> List[str]:
        seq_len = self.seq_len or max(len(tokens) for tokens in batch)
        return [tokens + [self.token_pad] * (seq_len - len(tokens)) for tokens in batch]

    def join(self, x: List[List[str]], ignore_special_tokens: bool = True) -> List[str]:
        self._ensure_batch(x)
        y = [self._joiner_fn(xi, ignore_special_tokens=ignore_special_tokens) for xi in x]
        return y
