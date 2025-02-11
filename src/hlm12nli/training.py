# Python Built-in Modules
from dataclasses import dataclass, field

# Third-Party Libraries
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

# Local Folders
from .data import NliDataset
from .encoding import Hlm12NliEncoder
from .tokenisation import Hlm12NliTokeniser


@dataclass(frozen=True)
class Hlm12NliHyperparameters:
    batch_size: int = 32
    max_epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 3


def train(
    tokeniser: Hlm12NliTokeniser,
    encoder: Hlm12NliEncoder,
    datasplit: NliDataset,
):
    data_loader = DataLoader(datasplit, batch_size=32, shuffle=True, num_workers=3)
    model = Hlm12NliEncoderTrainer(encoder=encoder)
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            ModelCheckpoint(monitor="train_loss"),
            TQDMProgressBar(),
        ],
    )
    trainer.fit(model, data_loader)


class Hlm12NliEncoderTrainer(pl.LightningModule):
    def __init__(self, encoder: Hlm12NliEncoder, margin=1.0):
        super(Hlm12NliEncoderTrainer, self).__init__()
        self.encoder = encoder
        self.loss = torch.nn.TripletMarginLoss()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_output = self.encoder(anchor)
        positive_output = self.encoder(positive)
        negative_output = self.encoder(negative)
        loss = self.loss(anchor_output, positive_output, negative_output)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
