# Python Built-in Modules
from dataclasses import dataclass

# Third-Party Libraries
import torch


@dataclass(frozen=True)
class Hlm12NliTokeniserOutput:
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
