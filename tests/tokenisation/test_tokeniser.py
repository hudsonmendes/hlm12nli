# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.tokenisation.config import Hlm12NliTextTokeniserConfig
from hlm12nli.tokenisation.tokeniser import Hlm12NliTokeniser


class Hlm12NliTokeniserIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "A", "test", "sentence", "##.", "Hudson", "##'s"]
        self.tokeniser = Hlm12NliTokeniser(Hlm12NliTextTokeniserConfig(vocab=self.vocab, do_lowercase=False))

    def test_tokenise_includes_start(self):
        test_sentence = "A test sentence"
        actual_output = self.tokeniser.tokenise([test_sentence])
        self.assertEqual(actual_output.input_ids[0][0].item(), self.tokeniser.vocab.get(self.tokeniser.token_str))

    def test_tokenise_includes_end(self):
        test_sentence = "A test sentence"
        actual_output = self.tokeniser.tokenise([test_sentence])
        self.assertEqual(actual_output.input_ids[0][-1].item(), self.tokeniser.vocab.get(self.tokeniser.token_end))

    def test_tokenise_includes_pad(self):
        test_input = ["A test sentence", "Hudson test"]
        actual_output = self.tokeniser.tokenise(test_input)
        self.assertEqual(actual_output.input_ids[1][-1].item(), self.tokeniser.vocab.get(self.tokeniser.token_pad))

    def test_tokenise_with_only_whitespace_splitters(self):
        test_sentence = "A test sentence"
        expected_input_ids = [self.tokeniser.vocab.get(t) for t in ["[STR]", "A", "test", "sentence", "[END]"]]
        actual_input_ids = self.tokeniser.tokenise([test_sentence]).input_ids
        self.assertEqual(actual_input_ids[0].tolist(), expected_input_ids)

    def test_tokenise_with_subwords(self):
        test_sentence = "Hudson's test"
        expected_input_ids = [self.tokeniser.vocab.get(t) for t in ["[STR]", "Hudson", "##'s", "test", "[END]"]]
        actual_input_ids = self.tokeniser.tokenise([test_sentence]).input_ids
        self.assertEqual(actual_input_ids[0].tolist(), expected_input_ids)

    def test_tokenise_with_unk_token(self):
        test_sentence = "A test sentence not_in_vocab"
        expected_input_ids = [self.tokeniser.vocab.get(t) for t in ["[STR]", "A", "test", "sentence", "[OOV]", "[END]"]]
        actual_input_ids = self.tokeniser.tokenise([test_sentence]).input_ids
        self.assertEqual(actual_input_ids[0].tolist(), expected_input_ids)

    def test_reverse_matches(self):
        test_input = ["A Hudson's test sentence.", "test"]
        tokenised_input_ids = self.tokeniser.tokenise(test_input).input_ids
        reversed_tokens = self.tokeniser.reverse(tokenised_input_ids)
        self.assertEqual(reversed_tokens, test_input)

    def test_reverse_with_special_tokens_matches(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_output = ["[STR] A Hudson's test sentence. [END]", "[STR] test [END] [PAD] [PAD] [PAD] [PAD] [PAD]"]
        batch_tokenized = self.tokeniser.tokenise(test_input).input_ids
        actual_output = self.tokeniser.reverse(batch_tokenized, ignore_special_tokens=False)
        self.assertEqual(actual_output, expected_output)

    def test_call_produces_correct_input_ids(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_input_ids = [[2, 4, 8, 9, 5, 6, 7, 3], [2, 5, 3, 0, 0, 0, 0, 0]]
        actual_input_ids = self.tokeniser(test_input).input_ids.tolist()
        self.assertEqual(actual_input_ids, expected_input_ids)

    def test_call_produces_correct_input_mask(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_input_mask = [
            [True, True, True, True, True, True, True, True],
            [True, True, True, False, False, False, False, False],
        ]
        actual_input_mask = self.tokeniser(test_input).input_mask.tolist()
        self.assertEqual(actual_input_mask, expected_input_mask)
