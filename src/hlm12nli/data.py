# Python Built-in Modules
from typing import Literal, Tuple, Union

# Third-Party Libraries
from datasets import load_dataset
from torch.utils.data import Dataset

# Local Folders
from .tokenisation import Hlm12NliTokenisation, Hlm12NliTokeniser


class NliDataset(Dataset):
    def __init__(
        self,
        anchors: Hlm12NliTokenisation,
        positives: Hlm12NliTokenisation,
        negatives: Hlm12NliTokenisation,
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self) -> int:
        return len(self.anchors.ids)

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[Hlm12NliTokenisation, Hlm12NliTokenisation, Hlm12NliTokenisation]:
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        negative = self.negatives[idx]
        return anchor, positive, negative


def load_nli_dataset(
    split: Literal["train", "validation", "test"],
    tokeniser: Hlm12NliTokeniser,
) -> NliDataset:
    data = load_dataset("snli")[split]

    # Filter to keep only entailment (0) and contradiction (2) examples
    entailment_data = data.filter(lambda ex: ex["label"] == 0)
    contradiction_data = data.filter(lambda ex: ex["label"] == 2)
    num_triplets = min(len(entailment_data), len(contradiction_data))

    # Build triplets:
    anchors_text = entailment_data["premise"][:num_triplets]
    positives_text = entailment_data["hypothesis"][:num_triplets]
    negatives_text = contradiction_data["hypothesis"][:num_triplets]

    return NliDataset(
        anchors=tokeniser(anchors_text),
        positives=tokeniser(positives_text),
        negatives=tokeniser(negatives_text),
    )
