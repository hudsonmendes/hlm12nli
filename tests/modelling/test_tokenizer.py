# Python Built-in Modules
import json
import pathlib
import unittest

# My Packages and Modules
from hlm12nli.modelling.tokenizer import Hlm12NliTokenizer


class TestHlm12NliTokenizer(unittest.TestCase):
    def setUp(self):
        self.vocab_file = pathlib.Path("test_vocab.json")
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "This": 3, "is": 4, "a": 5, "test": 6, "sentence": 7, ".": 8}
        with open(self.vocab_file, "w") as f:
            json.dump(self.vocab, f)
        self.tokenizer = Hlm12NliTokenizer(self.vocab_file)

    def tearDown(self):
        self.vocab_file.unlink()

    def test_load_vocab(self):
        expected_output = self.vocab
        self.assertEqual(self.tokenizer.load_vocab(self.vocab_file), expected_output)

    def test_tokenize_by_whitespace(self):
        test_sentence = "This is a test sentence."
        expected_output = ["This", "is", "a", "test", "sentence", "."]
        self.assertEqual(self.tokenizer._tokenize_by_whitespace(test_sentence), expected_output)

    def test_wordpiece_without_subwords(self):
        test_tokens = ["This", "is", "a", "test", "sentence", "."]
        expected_output = ["This", "is", "a", "test", "sentence", "."]
        self.assertEqual(self.tokenizer._wordpiece(test_tokens), expected_output)

    def test_wordpiece_with_subwords(self):
        test_tokens = ["Hudson's test"]
        expected_output = ["Hudson", "##s", "test"]
        self.assertEqual(self.tokenizer._wordpiece(test_tokens), expected_output)

    def test_wordpiece_with_unk_token(self):
        test_tokens = ["This", "is", "a", "test", "sentence", "not_in_vocab"]
        expected_output = ["This", "is", "a", "test", "sentence", "[UNK]"]
        self.assertEqual(self.tokenizer._wordpiece(test_tokens), expected_output)

    def test_tokenize(self):
        test_sentence = "This is a test sentence."
        expected_output = ["This", "is", "a", "test", "sentence", "."]
        self.assertEqual(self.tokenizer.tokenize(test_sentence), expected_output)
