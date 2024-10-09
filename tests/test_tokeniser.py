# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.tokenisation import Hlm12NliTextTokeniser, Hlm12NliTextTokeniserConfig


class TestTokeniser(unittest.TestCase):
    def setUp(self):
        self.tokeniser = Hlm12NliTextTokeniser(
            config=Hlm12NliTextTokeniserConfig(
                seqlen=12,
                vocab=[
                    "<start>",
                    "<end>",
                    "<pad>",
                    "<oov>",
                    "I",
                    "##'m",
                    "groot",
                    "##!",
                ],
            )
        )

    def test_call_matches_tokenise(self):
        x = "I'm groot!"
        self.assertListEqual(self.tokeniser(x=x).tokens, self.tokeniser.tokenise(x=x).tokens)
        self.assertListEqual(self.tokeniser(x=x).ids.tolist(), self.tokeniser.tokenise(x=x).ids.tolist())
        self.assertListEqual(self.tokeniser(x=x).mask.tolist(), self.tokeniser.tokenise(x=x).mask.tolist())

    def test_tokenise_tokens(self):
        x = "I'm groot bam!"
        y = [self.tokeniser.config.token_start, "I", "##'m", "groot", "bam", "##!", self.tokeniser.config.token_end]
        y += [self.tokeniser.config.token_pad] * (self.tokeniser.config.seqlen - len(y))
        self.assertListEqual(self.tokeniser.tokenise(x=[x]).tokens, [y])
        self.assertListEqual(self.tokeniser.tokenise(x=x).tokens, [y])

    def test_tokenise_ids(self):
        x = "I'm groot bam!"
        y = [0, 4, 5, 6, 3, 7, 1]
        y += [2] * (self.tokeniser.config.seqlen - len(y))
        self.assertListEqual(self.tokeniser.tokenise(x=[x]).ids.tolist(), [y])
        self.assertListEqual(self.tokeniser.tokenise(x=x).ids.tolist(), [y])

    def test_tokenise_mask(self):
        x = "I'm groot bam!"
        y = [True, True, True, True, True, True, True]
        y += [False] * (self.tokeniser.config.seqlen - len(y))
        self.assertListEqual(self.tokeniser.tokenise(x=[x]).mask.tolist(), [y])
        self.assertListEqual(self.tokeniser.tokenise(x=x).mask.tolist(), [y])
