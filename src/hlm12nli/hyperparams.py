# Python Built-in Modules
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hlm12NliEncoderHyperparams:
    embed_dim: int = field(default=128)
    lstm_dim: int = field(default=256)
    out_dim: int = field(default=3)
    lstm_bidirectional: bool = field(default=False)


@dataclass(frozen=True)
class Hlm12NliTokeniserHyperparams:
    vocab: list[str] = field(default_factory=list)
    seqlen: int = field(default=128)
    token_start: str = field(default="<start>")
    token_end: str = field(default="<end>")
    token_pad: str = field(default="<pad>")
    token_oov: str = field(default="<oov>")

    def __post_init__(self):
        special_tokens = (self.token_pad, self.token_oov, self.token_end, self.token_start)
        for token in special_tokens:
            self.vocab.insert(0, token)


@dataclass(frozen=True)
class Hlm12NliTrainingHyperparams:
    batch_size: int = field(default=32)
    max_epochs: int = field(default=100)
    learning_rate: float = field(default=1e-3)


@dataclass(frozen=True)
class Hlm12NliHyperparams:
    encoder: Hlm12NliEncoderHyperparams = field(default=Hlm12NliEncoderHyperparams())
    tokeniser: Hlm12NliTokeniserHyperparams = field(default=Hlm12NliTokeniserHyperparams())
    training: Hlm12NliTrainingHyperparams = field(default=Hlm12NliTrainingHyperparams())
