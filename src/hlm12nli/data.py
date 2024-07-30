# Third-Party Libraries
import torch
from torch.utils.data import Dataset


class NliDataset(Dataset):
    def __init__(self, anchors, positives, negatives):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        negative = self.negatives[idx]
        return anchor, positive, negative

    @staticmethod
    def load_sample() -> "NliDataset":
        return NliDataset(
            anchors=torch.randint(0, 50, (100, 50)),
            positives=torch.randint(0, 50, (100, 50)),
            negatives=torch.randint(0, 50, (100, 50)),
        )
