# Python Built-in Modules
from dataclasses import dataclass, field
from typing import Dict, List, Union

# Third-Party Libraries
import torch


@dataclass(frozen=True)
class Hlm12NliTextTokeniserConfig:
    vocab: List[str] = field(default_factory=list)
    seqlen: int = field(default=128)
    token_start: str = field(default="<start>")
    token_end: str = field(default="<end>")
    token_pad: str = field(default="<pad>")
    token_oov: str = field(default="<oov>")

    def __post_init__(self):
        special_tokens = (self.token_start, self.token_end, self.token_pad, self.token_oov)
        assert all(t in self.vocab for t in special_tokens), "All special tokens must be in the vocabulary"


@dataclass(frozen=True)
class Hlm12NliTextTokenisation:
    tokens: List[List[str]]
    ids: torch.IntTensor
    mask: torch.BoolTensor


class Hlm12NliTextTokeniser:
    config: Hlm12NliTextTokeniserConfig
    token_to_tid: Dict[str, int]

    def __init__(self, config: Hlm12NliTextTokeniserConfig):
        self.config = config
        self.token_to_tid = dict((token, tid) for (tid, token) in enumerate(config.vocab))
        self.tid_to_token = dict((tid, token) for (tid, token) in enumerate(config.vocab))

    def __call__(self, x: Union[str, List[str]]) -> Hlm12NliTextTokenisation:
        return self.tokenise(x=x)

    def tokenise(self, x: Union[str, List[str]]) -> Hlm12NliTextTokenisation:
        start, end = self.config.token_start, self.config.token_end
        pad, oov = self.config.token_pad, self.config.token_oov
        oovid = self.token_to_tid.get(oov)
        tokens = self._tokenise(x)
        tokens = [[start] + ts + [end] for ts in tokens]
        tokens = [ts + [pad] * (self.config.seqlen - len(ts)) for ts in tokens]
        tokens = [ts[: self.config.seqlen] for ts in tokens]
        return Hlm12NliTextTokenisation(
            tokens=tokens,
            ids=torch.IntTensor([[self.token_to_tid.get(t, oovid) for t in ts] for ts in tokens]),
            mask=torch.BoolTensor([[t != pad for t in ts] for ts in tokens]),
        )

    def detokenise(self, y: Union[List[str], List[List[str]], torch.IntTensor]) -> Union[str, List[str]]:
        if isinstance(y, torch.IntTensor):
            seqs: List[List[int]] = y.tolist()
            y = [[self.tid_to_token.get(yii, "") for yii in yi] for yi in seqs]
        return self._join(y=y)

    @staticmethod
    def _tokenise(x: Union[str, List[str]]) -> List[List[str]]:
        if not isinstance(x, list):
            x = [x if not isinstance(x, str) else str(x)]
        seqs = []
        for sent in x:
            tokens = []
            for token in sent.split():
                # collect all chars of token until it finds a non-alphanumeric char
                subtoken = ""
                for char in token:
                    if char.isalnum():
                        subtoken += char
                    else:
                        if subtoken:
                            tokens.append(subtoken)
                        subtoken = "##" + char
                tokens.append(subtoken)
            seqs.append(tokens)
        return seqs

    @staticmethod
    def _join(y: Union[List[str], List[List[str]]]) -> Union[str, List[str]]:
        single = False
        if isinstance(y, list) and len(y) > 0 and not isinstance(y[0], list):
            single = True
            y = [[yi if isinstance(yi, str) else str(yi) for yi in y]]

        seqs: List[str] = []
        for tokens in y:
            s = ""
            for token in tokens:
                if token.startswith("##"):
                    s += token[2:]
                else:
                    if s and not s.endswith(" "):
                        s += " "
                    s += token
            seqs.append(s)
        return seqs[0] if single else seqs
