# Python Built-in Modules
from dataclasses import dataclass

# Third-Party Libraries
import torch
from nest_ml.text.tokenisation import NestMLTextTokeniserOutputBase


@dataclass(frozen=True)
class Hlm12NliTokeniserOutput(NestMLTextTokeniserOutputBase):
    """
    Represents the output of the Hlm12NliTokeniser, including
    tokens, padded_tokens, encoded_tokens, and mask.
    """

    def encoded_tokens_to_tensor(self, device: torch.device = None) -> torch.LongTensor:
        """
        Requires encoded_tokens to be already padded and converts the encoded_tokens to a tensor.

        Args:
            device: torch.device
                The device to move the tensor to.
        """
        tensor = torch.LongTensor(self.encoded_tokens)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def mask_to_tensor(self, device: torch.device = None) -> torch.BoolTensor:
        """
        Requires mask to be already padded and converts the mask to a tensor.

        Args:
            device: torch.device
                The device to move the tensor to.
        """
        tensor = torch.BoolTensor(self.mask)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
