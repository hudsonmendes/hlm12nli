# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.tokenisation.config import Hlm12NliTextTokeniserConfig
from hlm12nli.tokenisation.tokeniser import Hlm12NliTokeniser


class IntegrationTestHlm12NliTokeniser(unittest.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "A", "test", "sentence", "##.", "Hudson", "##'s"]
        self.tokeniser = Hlm12NliTokeniser(Hlm12NliTextTokeniserConfig(vocab=self.vocab, do_lowercase=False))

    def test_tokenise_includes_start(self):
        test_sentence = "A test sentence"
        actual_output = self.tokeniser.tokenise([test_sentence])
        self.assertEqual(actual_output.padded_tokens[0][0], self.tokeniser.token_str)
        self.assertEqual(actual_output.encoded_tokens[0][0], self.vocab.index(self.tokeniser.token_str))

    def test_tokenise_includes_end(self):
        test_sentence = "A test sentence"
        actual_output = self.tokeniser.tokenise([test_sentence])
        self.assertEqual(actual_output.padded_tokens[0][-1], self.tokeniser.token_end)
        self.assertEqual(actual_output.encoded_tokens[0][-1], self.tokeniser.vocab.get(self.tokeniser.token_end))

    def test_tokenise_includes_pad(self):
        test_input = ["A test sentence", "Hudson test"]
        actual_output = self.tokeniser.tokenise(test_input)
        self.assertEqual(actual_output.padded_tokens[1][-1], self.tokeniser.token_pad)
        self.assertEqual(actual_output.encoded_tokens[1][-1], self.tokeniser.vocab.get(self.tokeniser.token_pad))

    def test_tokenise_with_only_whitespace_splitters(self):
        test_sentence = "A test sentence"
        expected = [self.tokeniser.vocab.get(t) for t in ["[STR]", "A", "test", "sentence", "[END]"]]
        actual = self.tokeniser.tokenise([test_sentence]).encoded_tokens
        self.assertEqual(actual[0], expected)

    def test_tokenise_with_subwords(self):
        test_sentence = "Hudson's test"
        expected = [self.tokeniser.vocab.get(t) for t in ["[STR]", "Hudson", "##'s", "test", "[END]"]]
        actual = self.tokeniser.tokenise([test_sentence]).encoded_tokens
        self.assertEqual(actual[0], expected)

    def test_tokenise_with_unk_token(self):
        test_sentence = "A test sentence not_in_vocab"
        expected = [self.tokeniser.vocab.get(t) for t in ["[STR]", "A", "test", "sentence", "[OOV]", "[END]"]]
        actual = self.tokeniser.tokenise([test_sentence]).encoded_tokens
        self.assertEqual(actual[0], expected)

    def test_reverse_matches(self):
        test_input = ["A Hudson's test sentence.", "test"]
        actual = self.tokeniser.tokenise(test_input).encoded_tokens
        reversed = self.tokeniser.reverse(actual)
        self.assertEqual(reversed, test_input)

    def test_reverse_with_special_tokens_matches(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected = ["[STR] A Hudson's test sentence. [END]", "[STR] test [END] [PAD] [PAD] [PAD] [PAD] [PAD]"]
        actual = self.tokeniser.tokenise(test_input).encoded_tokens
        reversed = self.tokeniser.reverse(actual, ignore_special_tokens=False)
        self.assertEqual(reversed, expected)

    def test_call_produces_correct_input_ids(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected = [[2, 4, 8, 9, 5, 6, 7, 3], [2, 5, 3, 0, 0, 0, 0, 0]]
        actual = self.tokeniser(test_input).encoded_tokens
        self.assertEqual(actual, expected)

    def test_call_produces_correct_input_mask(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected = [
            [True, True, True, True, True, True, True, True],
            [True, True, True, False, False, False, False, False],
        ]
        actual = self.tokeniser(test_input).mask
        self.assertEqual(actual, expected)
