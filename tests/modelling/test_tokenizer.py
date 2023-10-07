# Python Built-in Modules
import json
import pathlib
import unittest

# My Packages and Modules
from hlm12nli.modelling.tokenizer import Hlm12NliTokenizer


class TestHlm12NliTokenizer(unittest.TestCase):
    def setUp(self):
        vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "This", "is", "a", "test", "sentence", "##.", "Hudson", "s"]
        self.tokenizer = Hlm12NliTokenizer(
            vocab={token: i for i, token in enumerate(vocab)},
            do_lowercase=False,
        )

    def tearDown(self):
        self.vocab_file.unlink()

    def test_tokenize_returns_single_or_list_depending_on_input(self):
        self.assertIsInstance(self.tokenizer.tokenize(["a", "b"])[0], list)
        self.assertIsInstance(self.tokenizer.tokenize("a")[0], str)

    def test_tokenize_lowercases(self):
        vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "this", "is", "a", "test", "sentence", "##.", "Hudson", "s"]
        tokenizer = Hlm12NliTokenizer(
            vocab={token: i for i, token in enumerate(vocab)},
            do_lowercase=False,
        )
        test_sentence = "This is a test sentence."
        expected_output = ["[STR]", "this", "is", "a", "test", "sentence", "##.", "[END]"]
        self.assertEqual(tokenizer.tokenize(test_sentence), expected_output)

    def test_tokenize_includes_start(self):
        test_sentence = "This is a test sentence."
        expected_output = ["[STR]", "This", "is", "a", "test", "sentence", "##.", "[END]"]
        self.assertEqual(self.tokenizer.tokenize(test_sentence), expected_output)

    def test_tokenize_includes_end(self):
        test_sentence = "This is a test sentence."
        expected_output = ["[STR]", "This", "is", "a", "test", "sentence", "##.", "[END]"]
        self.assertEqual(self.tokenizer.tokenize(test_sentence), expected_output)

    def test_tokenize_includes_pad(self):
        test_sentence = ["This is a test sentence.", "Hudson's test"]
        expected_output = [
            ["[START]", "This", "is", "a", "test", "sentence", "##.", "[END]"],
            ["[START]", "Hudson", "##s", "test", "[END]", "[PAD]", "[PAD]", "[PAD]"],
        ]
        self.assertEqual(self.tokenizer.tokenize(test_sentence), expected_output)

    def test_tokenize_with_only_whitespace_splitters(self):
        test_sentence = "This is a test sentence."
        expected_output = ["[STR]", "This", "is", "a", "test", "sentence", "##.", "[END]"]
        self.assertEqual(self.tokenizer.tokenize(test_sentence), expected_output)

    def test_tokenize_with_subwords(self):
        test_tokens = ["Hudson's test"]
        expected_output = ["[STR]", "Hudson", "##s", "test", "[END]"]
        self.assertEqual(self.tokenizer.tokenize(test_tokens), expected_output)

    def test_tokenize_with_unk_token(self):
        test_tokens = ["This", "is", "a", "test", "sentence", "not_in_vocab"]
        expected_output = ["[STR]", "This", "is", "a", "test", "sentence", "[UNK]", "[END]"]
        self.assertEqual(self.tokenizer.tokenize(test_tokens), expected_output)
