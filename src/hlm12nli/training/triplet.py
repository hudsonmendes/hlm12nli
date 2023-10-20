# Python Built-in Modules
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Third-Party Libraries
import lightning.pytorch as pl
import torch

# My Packages and Modules
from hlm12nli.modelling import Hlm12NliEncoder
from hlm12nli.tokenisation import Hlm12NliTokeniserOutput


@dataclass(frozen=True)
class Hlm12NliTripletConfig:
    learning_rate: float = field()
    triplet_loss_margin: float = field()
    triplet_loss_p: int = field()


class Hlm12NliTripletModel(pl.LightningModule):
    def __init__(self, config: Hlm12NliTripletConfig, encoder: Hlm12NliEncoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        self.loss = torch.nn.TripletMarginLoss(margin=config.triplet_loss_margin, p=config.triplet_loss_p)

    def training_step(
        self,
        batch: Tuple[Hlm12NliTokeniserOutput, Hlm12NliTokeniserOutput, Hlm12NliTokeniserOutput],
        batch_idx: int,
    ) -> Dict[str, torch.FloatTensor]:
        anchor, positive, negative = batch

        # forward pass
        anchor_embedding = self.encoder(input_ids=anchor.input_ids, input_mask=anchor.input_mask)
        positive_embedding = self.encoder(input_ids=positive.input_ids, input_mask=positive.input_mask)
        negative_embedding = self.encoder(input_ids=negative.input_ids, input_mask=negative.input_mask)

        # calculate triplet loss
        loss = self.loss(anchor_embedding, positive_embedding, negative_embedding)

        # logging
        self.log("train_loss", loss)

        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer
