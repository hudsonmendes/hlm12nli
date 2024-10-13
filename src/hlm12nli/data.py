# Python Built-in Modules
from typing import Literal, Tuple, Union

# Third-Party Libraries
from datasets import load_dataset
from torch.utils.data import Dataset

# Local Folders
from .tokenisation import Hlm12NliTextTokenisation, Hlm12NliTextTokeniser


class NliDataset(Dataset):
    def __init__(
        self,
        anchors: Hlm12NliTextTokenisation,
        positives: Hlm12NliTextTokenisation,
        negatives: Hlm12NliTextTokenisation,
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[Hlm12NliTextTokenisation, Hlm12NliTextTokenisation, Hlm12NliTextTokenisation]:
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        negative = self.negatives[idx]
        return anchor, positive, negative


def load_split(
    split: Literal["train", "validation", "test"],
    tokeniser: Hlm12NliTextTokeniser,
) -> NliDataset:
    # TODO: sample triplets from SNLI
    data = load_dataset("snli")[split]
    return NliDataset(
        anchors=tokeniser(),
        positives=tokeniser(data["hypothesis"]),
        negatives=tokeniser(data["premise"]),
    )
