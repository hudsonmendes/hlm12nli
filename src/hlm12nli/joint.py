# Third-Party Libraries
import pytorch_lightning as pl
import torch

# Local Folders
from .encoder import Hlm12NliEncoder


class Hlm12NliJoint(pl.LightningModule):
    def __init__(self, encoder: Hlm12NliEncoder, margin=1.0):
        super(Hlm12NliJoint, self).__init__()
        self.encoder = encoder
        self.criterion = torch.nn.TripletMarginLoss()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_output = self.encoder(anchor)
        positive_output = self.encoder(positive)
        negative_output = self.encoder(negative)
        loss = self.criterion(anchor_output, positive_output, negative_output)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
