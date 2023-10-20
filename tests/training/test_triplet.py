# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12nli.modelling.encoder import Hlm12NliEncoder, Hlm12NliEncoderConfig
from hlm12nli.tokenisation.tokeniser import Hlm12NliTextTokeniserConfig, Hlm12NliTokeniser
from hlm12nli.training.triplet import Hlm12NliTripletConfig, Hlm12NliTripletModel


class IntegrationTestHlm12NliTripletModel(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Hlm12NliTokeniser(
            config=Hlm12NliTextTokeniserConfig(
                vocab=[
                    "[PAD]",
                    "[OOV]",
                    "[STR]",
                    "[END]",
                    "this",
                    "is",
                    "a",
                    "sentence",
                    "yes",
                    "no",
                    "correct",
                    "right",
                    "wrong",
                    "first",
                    "second",
                    "third",
                    "fourth",
                    "##,",
                    "##.",
                    "##'re",
                ],
            ),
        )
        self.encoder = Hlm12NliEncoder(
            config=Hlm12NliEncoderConfig(
                vocab_size=self.tokenizer.config.vocab_size,
                token_vec_dims=16,
                token_id_pad=0,
                hidden_dims=16,
                hidden_bidir=True,
                attn_heads=2,
                attn_dropout=0.1,
                output_dims=8,
            )
        )
        self.config = Hlm12NliTripletConfig(
            learning_rate=5e-3,
            triplet_loss_margin=1.0,
            triplet_loss_p=2,
        )
        self.model = Hlm12NliTripletModel(
            config=self.config,
            encoder=self.encoder,
        )

    def test_training_step_calculates_loss(self):
        x = self._x()
        output = self.model.training_step(x, 0)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)

    def _x(self):
        anchor = self.tokenizer.tokenise(
            [
                "This is a test sentence",
                "This is another test sentence",
                "This is a third test sentence",
            ]
        )
        positive = self.tokenizer.tokenise(
            [
                "yes, this is a test sentence",
                "you're right, this is another test sentence",
                "correct, this is a third test sentence",
            ]
        )
        negative = self.tokenizer.tokenise(
            [
                "this is not a test sentence",
                "this is the same test sentence",
                "this is a fourth test sentence",
            ]
        )

        return anchor, positive, negative
