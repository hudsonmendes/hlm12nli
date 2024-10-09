# Python Built-in Modules
from typing import Tuple

# Third-Party Libraries
from torch.utils.data import Dataset

# Local Folders
from .tokenisation import Hlm12NliTextTokenisation


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

    def __getitem__(self, idx: int) -> Tuple[Hlm12NliTextTokenisation, Hlm12NliTextTokenisation, Hlm12NliTextTokenisation]:
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        negative = self.negatives[idx]
        return anchor, positive, negative
