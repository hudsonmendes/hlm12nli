# Python Built-in Modules
from dataclasses import dataclass
from typing import List

# Third-Party Libraries
import torch


@dataclass(frozen=True)
class Hlm12NliConfig:
    vocab: List[str]
    dim_embed: int
    dim_lstm: int
    dim_out: int
