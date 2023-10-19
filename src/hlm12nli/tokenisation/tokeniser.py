# Python Built-in Modules
from typing import Dict, List, Optional

# Third-Party Libraries
import torch

# Local Folders
from .config import Hlm12NliTextTokeniserConfig
from .joining import Hlm12NliTokeniserJoiner
from .output import Hlm12NliTokeniserOutput
from .splitter import Hlm12NliTokeniserSplitter


class Hlm12NliTokeniser:
    """
    Tokeniser for the Hlm12Nli model, implemented using the wordpiece algorithm
    with padding, truncating and special tokens capabilities.
    Inherits from `NestMLTextTokeniserBase`.

    Attributes:
        config: Hlm12NliTextTokeniserConfig
            The tokeniser configuration.
        device: torch.device | None
            The device to be used for the tokeniser, defaults to None ("cpu")
    """

    config: Hlm12NliTextTokeniserConfig
    vocab_by_token: Dict[str, int]
    vocab_by_token_id: Dict[int, str]
    device: Optional[torch.device]

    def __init__(self, config: Hlm12NliTextTokeniserConfig, device: Optional[torch.device] = None):
        """
        Constructs a new Hlm12NliTokeniser.

        Args:
            config: Hlm12NliTextTokeniserConfig
                The tokeniser configuration.
        """
        self.config = config
        self.vocab_by_token = config.vocab_by_token
        self.vocab_by_token_id = config.vocab_by_token_id
        self.device = device
        self.splitter_fn = Hlm12NliTokeniserSplitter(
            config.vocab,
            token_oov=config.token_oov,
            expr_subword=config.expr_subword,
        )
        self.joiner_fn = Hlm12NliTokeniserJoiner(
            expr_subword=config.expr_subword,
        )

    def tokenise(self, x: List[List[str]]) -> Hlm12NliTokeniserOutput:
        """
        Tokenises a batch of examples.

        Args:
            x: List[List[str]]
                The batch of examples to be tokenised.

        Returns:
            Hlm12NliTokeniserOutput
                The tokenised batch of examples.
        """
        seqs = self.splitter_fn(x)
        wrapped = [[self.config.token_str] + seq + [self.config.token_end] for seq in seqs]
        seq_len = self.config.seq_len or max(len(tokens) for tokens in wrapped)
        padded = [tokens + [self.config.token_pad] * (seq_len - len(tokens)) for tokens in wrapped]
        truncated = [tokens[: self.config.max_seq_len] for tokens in padded]
        mask = [[t != self.config.token_pad for t in seq] for seq in truncated]
        token_id_oov = self.vocab_by_token[self.config.token_oov]
        translated = [[self.vocab_by_token.get(token, token_id_oov) for token in tokens] for tokens in truncated]
        return Hlm12NliTokeniserOutput(
            input_ids=torch.LongTensor(translated).to(self.device),
            input_mask=torch.BoolTensor(mask).to(self.device),
        )

    def reverse(self, x: torch.LongTensor, ignore_special_tokens: bool = True) -> List[str]:
        """
        Reverses a batch of tokens into what should be the original example,
        removing special tokens if `ignore_special_tokens` is True.

        Args:
            x: List[List[int]]]
                The batch of encoded tokens (vocabulary indices) to be reversed.
            ignore_special_tokens: bool
                Whether to ignore special tokens.

        Returns:
            List[str]
                The original batch of examples.
        """
        y = [
            [self.vocab_by_token_id.get(token_id.item(), self.config.token_oov) for token_id in token_ids]
            for token_ids in x
        ]
        if ignore_special_tokens:
            y = [[t for t in tokens if t not in self.config.special_tokens] for tokens in y]
        y = self.joiner_fn(y)
        return y
