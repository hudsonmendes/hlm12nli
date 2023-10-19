# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12nli.modelling.config import Hlm12NliConfig
from hlm12nli.modelling.encoder import Hlm12NliEncoder
from hlm12nli.tokenisation import Hlm12NliTokeniserOutput


class IntegrationTestHlm12NliEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = Hlm12NliEncoder(
            config=Hlm12NliConfig(
                vocab_size=10,
                token_vec_dims=16,
                token_id_pad=0,
                hidden_dims=16,
                hidden_bidir=True,
                attn_heads=2,
                attn_dropout=0.1,
                output_dims=2,
            )
        )

    def test_forward_produces_correct_representation_shape(self):
        x = self._x()
        y = self.encoder.forward(input_ids=x.input_ids, input_mask=x.input_mask)
        self.assertEqual(y.shape, (x.input_ids.shape[0], self.encoder.config.output_dims))

    def _x(self):
        return Hlm12NliTokeniserOutput(
            input_ids=torch.LongTensor(
                [
                    [1, 3, 4, 5, 6, 2],
                    [1, 6, 2, 0, 0, 0],
                ]
            ),
            input_mask=torch.BoolTensor(
                [
                    [True, True, True, True, True, True],
                    [True, True, True, False, False, False],
                ]
            ),
        )
