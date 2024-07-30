# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12nli.encoder import Hlm12NliConfig, Hlm12NliEncoder


class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = Hlm12NliEncoder(
            config=Hlm12NliConfig(
                vocab=["[PAD]", "[OOV]", "[CLS]", "I", " am", " groot", " not", "!"],
                dim_embed=3,
                dim_lstm=16,
                dim_out=2,
            )
        )

    def test_forward_outputs_correct_shape(self):
        y = self.subject.forward(x=torch.tensor([[2, 3, 4, 5, 7, 2, 0, 0], [2, 1, 3, 4, 6, 5, 7, 2]]))
        self.assertEqual(y.shape, (2, 2))
