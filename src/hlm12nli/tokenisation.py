# Python Built-in Modules
from dataclasses import dataclass
from typing import Union

# Third-Party Libraries
import torch

# Local Folders
from .hyperparams import Hlm12NliHyperparams, Hlm12NliTokeniserHyperparams


@dataclass(frozen=True)
class Hlm12NliTokenisation:
    tokens: Union[list[list[str]], list[str]]
    ids: torch.IntTensor
    mask: torch.BoolTensor

    def __getitem__(self, idx: Union[int, slice]) -> "Hlm12NliTokenisation":
        return Hlm12NliTokenisation(
            tokens=self.tokens[idx],
            ids=self.ids[idx].type(torch.IntTensor),
            mask=self.mask[idx].type(torch.BoolTensor),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.tokens), len(self.tokens[0])


class Hlm12NliTokeniser:
    hyperparams: Hlm12NliTokeniserHyperparams
    token_to_tid: dict[str, int]
    tid_to_token: dict[int, str]

    def __init__(self, hyperparams: Hlm12NliHyperparams):
        self.hyperparams = hyperparams.tokeniser
        self.token_to_tid = dict((token, tid) for (tid, token) in enumerate(self.hyperparams.vocab))
        self.tid_to_token = dict((tid, token) for (tid, token) in enumerate(self.hyperparams.vocab))

    def __call__(self, x: Union[str, list[str]]) -> Hlm12NliTokenisation:
        return self.tokenise(x=x)

    def tokenise(self, x: Union[str, list[str]]) -> Hlm12NliTokenisation:
        start, end = self.hyperparams.token_start, self.hyperparams.token_end
        pad, oov = self.hyperparams.token_pad, self.hyperparams.token_oov
        oovid = self.token_to_tid.get(oov)
        tokens = self._tokenise(x)
        tokens = [[start] + ts + [end] for ts in tokens]
        tokens = [ts + [pad] * (self.hyperparams.seqlen - len(ts)) for ts in tokens]
        tokens = [ts[: self.hyperparams.seqlen] for ts in tokens]
        return Hlm12NliTokenisation(
            tokens=tokens,
            ids=torch.IntTensor([[self.token_to_tid.get(t.lower(), oovid) for t in ts] for ts in tokens]),
            mask=torch.BoolTensor([[t != pad for t in ts] for ts in tokens]),
        )

    def detokenise(self, y: Union[list[str], list[list[str]], torch.IntTensor, Hlm12NliTokenisation]) -> Union[str, list[str]]:
        if isinstance(y, Hlm12NliTokenisation):
            y = y.ids
        if isinstance(y, torch.IntTensor):
            seqs: list[list[int]] = y.tolist()
            if isinstance(seqs, list) and len(seqs) > 0 and isinstance(seqs[0], int):
                seqs = [seqs]
            y = [[self.tid_to_token.get(yii, "") for yii in yi] for yi in seqs]
        return self._join(y=y, ignored_tokens=self.hyperparams.special_tokens)

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
    def _join(y: Union[list[str], list[list[str]]], ignored_tokens: tuple[str, ...]) -> Union[str, list[str]]:
        single = False
        if isinstance(y, list) and len(y) > 0 and not isinstance(y[0], list):
            single = True
            y = [[yi if isinstance(yi, str) else str(yi) for yi in y]]

        seqs: list[str] = []
        for tokens in y:
            s = ""
            for token in tokens:
                if token in ignored_tokens:
                    continue
                elif token.startswith("##"):
                    s += token[2:]
                else:
                    if s and not s.endswith(" "):
                        s += " "
                    s += token
            seqs.append(s)
        return seqs[0] if single else seqs
