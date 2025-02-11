# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.encoding import Hlm12NliEncoder
from hlm12nli.hyperparams import Hlm12NliEncoderHyperparams, Hlm12NliHyperparams, Hlm12NliTokeniserHyperparams, Hlm12NliTrainingHyperparams
from hlm12nli.tokenisation import Hlm12NliTokeniser
from hlm12nli.training import train


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.hyperparams = Hlm12NliHyperparams(
            tokeniser=Hlm12NliTokeniserHyperparams(vocab=["a", "b", "c", "d", "e"]),
            encoder=Hlm12NliEncoderHyperparams(dim_embed=4, dim_lstm=8, dim_out=2),
            training=Hlm12NliTrainingHyperparams(n_epochs=1, batch_size=1, lr=0.01),
        )

    def test_train_updates_parameters(self):
        tokeniser = Hlm12NliTokeniser(hyperparams=self.hyperparams)
        encoder = Hlm12NliEncoder(hyperparams=self.hyperparams)
        before = [p.clone() for p in encoder.parameters()]
        train(encoder=encoder, tokeniser=tokeniser)
        self.assertNotEqual(before, encoder.parameters())
