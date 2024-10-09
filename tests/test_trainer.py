# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.encoding import Hlm12NliConfig, Hlm12NliEncoder
from hlm12nli.training import train


class TestTrainer(unittest.TestCase):
    def test_train_updates_parameters(self):
        encoder = Hlm12NliEncoder(config=Hlm12NliConfig(n_vocab=5, dim_embed=4, dim_lstm=8, dim_out=2))
        before = [p.clone() for p in encoder.parameters()]
        train(encoder=encoder)
        self.assertNotEqual(before, encoder.parameters())
