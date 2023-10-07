# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.tokenisation.tokeniser import (
    Hlm12NliTokeniser,
    Hlm12NliTokeniserBatchRequiredError,
    Hlm12NliTokeniserSeqLenError,
    Hlm12NliTokeniserVocabTokenCasingError,
)


class Hlm12NliTokeniserIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "A", "test", "sentence", "##.", "Hudson", "##'s"]
        self.tokeniser = Hlm12NliTokeniser(vocab=self.vocab, do_lowercase=False)

    def test_ctor_accepts_vocab_as_dict_or_list(self):
        t1 = Hlm12NliTokeniser(vocab=self.vocab, do_lowercase=False)
        t2 = Hlm12NliTokeniser(vocab={token: i for i, token in enumerate(self.vocab)}, do_lowercase=False)
        self.assertEqual(t1.vocab, t2.vocab)

    def test_ctor_fails_if_seq_len_greater_than_max_seq_len(self):
        with self.assertRaises(Hlm12NliTokeniserSeqLenError):
            Hlm12NliTokeniser(vocab=self.vocab, seq_len=1025, max_seq_len=1024)

    def test_ctor_fails_if_token_not_lowercase(self):
        with self.assertRaises(Hlm12NliTokeniserVocabTokenCasingError):
            Hlm12NliTokeniser(vocab=self.vocab + ["NotLowercase"], do_lowercase=True)

    def test_split_requires_batch(self):
        with self.assertRaises(Hlm12NliTokeniserBatchRequiredError):
            self.tokeniser.split("A test sentence")

    def test_split_lowercases(self):
        vocab = [t.lower() if not t.startswith("[") else t for t in self.vocab]
        tokeniser = Hlm12NliTokeniser(vocab=vocab, do_lowercase=True)
        test_sentence = "A test sentence"
        expected_output = ["[STR]", "a", "test", "sentence", "[END]"]
        self.assertListEqual(tokeniser.split([test_sentence])[0], expected_output)

    def test_split_includes_start(self):
        test_sentence = "A test sentence"
        self.assertEqual(self.tokeniser.split([test_sentence])[0][0], self.tokeniser.token_str)

    def test_split_includes_end(self):
        test_sentence = "A test sentence"
        self.assertEqual(self.tokeniser.split([test_sentence])[0][-1], self.tokeniser.token_end)

    def test_split_includes_pad(self):
        test_sentence = ["A test sentence", "Hudson test"]
        expected_output = [
            ["[STR]", "A", "test", "sentence", "[END]"],
            ["[STR]", "Hudson", "test", "[END]", "[PAD]"],
        ]
        actual_output = self.tokeniser.split(test_sentence)
        self.assertListEqual(actual_output, expected_output)

    def test_split_with_only_whitespace_splitters(self):
        test_sentence = "A test sentence"
        expected_output = ["[STR]", "A", "test", "sentence", "[END]"]
        self.assertEqual(self.tokeniser.split([test_sentence])[0], expected_output)

    def test_split_with_subwords(self):
        test_sentence = "Hudson's test"
        expected_output = ["[STR]", "Hudson", "##'s", "test", "[END]"]
        self.assertEqual(self.tokeniser.split([test_sentence])[0], expected_output)

    def test_split_with_unk_token(self):
        test_sentence = "A test sentence not_in_vocab"
        expected_output = ["[STR]", "A", "test", "sentence", "[OOV]", "[END]"]
        self.assertEqual(self.tokeniser.split([test_sentence])[0], expected_output)

    def test_split_and_join_without_special_tokens_match(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_output = ["A Hudson's test sentence.", "test"]
        batch_tokenized = self.tokeniser.split(test_input)
        actual_output = self.tokeniser.join(batch_tokenized)
        self.assertEqual(actual_output, expected_output)

    def test_split_and_join_with_special_tokens_match(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_output = ["[STR] A Hudson's test sentence. [END]", "[STR] test [END] [PAD] [PAD] [PAD] [PAD] [PAD]"]
        batch_tokenized = self.tokeniser.split(test_input)
        actual_output = self.tokeniser.join(batch_tokenized, ignore_special_tokens=False)
        self.assertEqual(actual_output, expected_output)

    def test_encode_produces_correct_input_ids(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_input_ids = [[3, 4, 8, 9, 10, 7], [3, 5, 7, 0, 0, 0, 0, 0]]
        actual_input_ids = self.tokeniser(test_input)["input_ids"]
        self.assertEqual(actual_input_ids, expected_input_ids)

    def test_encode_produces_correct_attention_mask(self):
        test_input = ["A Hudson's test sentence.", "test"]
        expected_input_ids = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0]]
        actual_input_ids = self.tokeniser(test_input)["input_ids"]
        self.assertEqual(actual_input_ids, expected_input_ids)