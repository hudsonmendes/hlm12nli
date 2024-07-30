# Third-Party Libraries
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

# Local Folders
from .config import Hlm12NliConfig
from .data import NliDataset
from .encoder import Hlm12NliEncoder
from .joint import Hlm12NliJoint


def train() -> Hlm12NliEncoder:
    config = Hlm12NliConfig()
    encoder = Hlm12NliEncoder(config)

    dataset = NliDataset.load_sample()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Hlm12NliJoint(encoder=encoder)

    trainer = pl.Trainer(max_epochs=10, callbacks=[ModelCheckpoint(monitor="train_loss"), TQDMProgressBar()])
    trainer.fit(model, data_loader)
