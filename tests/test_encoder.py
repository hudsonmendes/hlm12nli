# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12nli.encoding import Hlm12NliConfig, Hlm12NliEncoder
from hlm12nli.tokenisation import Hlm12NliTextTokenisation


class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = Hlm12NliEncoder(
            config=Hlm12NliConfig(
                n_vocab=10,
                dim_embed=3,
                dim_lstm=16,
                dim_out=2,
            )
        )

    def test_forward_outputs_1_by_dim_out(self):
        y = self.subject.forward(
            x=Hlm12NliTextTokenisation(
                tokens=[["<start>", "I", "##'m", "groot", "bam", "##!", "<end>"]],
                ids=torch.IntTensor([[0, 4, 5, 6, 3, 7, 1]]),
                mask=torch.BoolTensor([[True, True, True, True, True, True, True]]),
            )
        )
        self.assertEqual(y.shape, (1, 2))

    def test_forward_outputs_2_by_dim_out(self):
        y = self.subject.forward(
            x=Hlm12NliTextTokenisation(
                tokens=[
                    ["<start>", "I", "##'m", "groot", "bam", "##!", "<end>"],
                    ["<start>", "hey", "##!", "<end>", "<pad>", "<pad>", "<pad>"],
                ],
                ids=torch.IntTensor(
                    [
                        [0, 4, 5, 6, 3, 7, 1],
                        [0, 3, 5, 1, 2, 2, 2],
                    ]
                ),
                mask=torch.BoolTensor(
                    [
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, False, False, False],
                    ]
                ),
            )
        )
        self.assertEqual(y.shape, (2, 2))
