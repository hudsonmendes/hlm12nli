# Python Built-in Modules
from dataclasses import dataclass
from typing import Union

# Third-Party Libraries
import torch

# Local Folders
from .hyperparams import Hlm12NliHyperparams


@dataclass(frozen=True)
class Hlm12NliTextTokenisation:
    tokens: list[list[str]]
    ids: torch.IntTensor
    mask: torch.BoolTensor


class Hlm12NliTextTokeniser:
    hyperparams: Hlm12NliHyperparams
    token_to_tid: dict[str, int]
    tid_to_token: dict[int, str]

    def __init__(self, hyperparams: Hlm12NliHyperparams):
        self.hyperparams = hyperparams
        self.token_to_tid = dict((token, tid) for (tid, token) in enumerate(hyperparams.tokeniser.vocab))
        self.tid_to_token = dict((tid, token) for (tid, token) in enumerate(hyperparams.tokeniser.vocab))

    def __call__(self, x: Union[str, list[str]]) -> Hlm12NliTextTokenisation:
        return self.tokenise(x=x)

    def tokenise(self, x: Union[str, list[str]]) -> Hlm12NliTextTokenisation:
        start, end = self.hyperparams.tokeniser.token_start, self.hyperparams.tokeniser.token_end
        pad, oov = self.hyperparams.tokeniser.token_pad, self.hyperparams.tokeniser.token_oov
        oovid = self.token_to_tid.get(oov)
        tokens = self._tokenise(x)
        tokens = [[start] + ts + [end] for ts in tokens]
        tokens = [ts + [pad] * (self.hyperparams.tokeniser.seqlen - len(ts)) for ts in tokens]
        tokens = [ts[: self.hyperparams.tokeniser.seqlen] for ts in tokens]
        return Hlm12NliTextTokenisation(
            tokens=tokens,
            ids=torch.IntTensor([[self.token_to_tid.get(t, oovid) for t in ts] for ts in tokens]),
            mask=torch.BoolTensor([[t != pad for t in ts] for ts in tokens]),
        )

    def detokenise(self, y: Union[list[str], list[list[str]], torch.IntTensor]) -> Union[str, list[str]]:
        if isinstance(y, torch.IntTensor):
            seqs: list[list[int]] = y.tolist()
            y = [[self.tid_to_token.get(yii, "") for yii in yi] for yi in seqs]
        return self._join(y=y)

    @staticmethod
    def _tokenise(x: Union[str, list[str]]) -> list[list[str]]:
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
    def _join(y: Union[list[str], list[list[str]]]) -> Union[str, list[str]]:
        single = False
        if isinstance(y, list) and len(y) > 0 and not isinstance(y[0], list):
            single = True
            y = [[yi if isinstance(yi, str) else str(yi) for yi in y]]

        seqs: list[str] = []
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
