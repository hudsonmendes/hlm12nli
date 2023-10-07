# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.modelling.tokenizer import Hlm12NliTokenizer, Hlm12NliTokenizerBatchRequiredError


class TestHlm12NliTokenizer(unittest.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "A", "test", "sentence", "##.", "Hudson", "##'s"]
        self.tokenizer = Hlm12NliTokenizer(vocab=self.vocab, do_lowercase=False)

    def test_ctor_accepts_vocab_as_dict_or_list(self):
        t1 = Hlm12NliTokenizer(vocab=self.vocab, do_lowercase=False)
        t2 = Hlm12NliTokenizer(vocab={token: i for i, token in enumerate(self.vocab)}, do_lowercase=False)
        self.assertEqual(t1.vocab, t2.vocab)

    def test_tokenize_requires_batch(self):
        with self.assertRaises(Hlm12NliTokenizerBatchRequiredError):
            self.tokenizer.tokenize("A test sentence")

    def test_tokenize_lowercases(self):
        vocab = [t.lower() if not t.startswith("[") else t for t in self.vocab]
        tokenizer = Hlm12NliTokenizer(vocab=vocab, do_lowercase=True)
        test_sentence = "A test sentence"
        expected_output = ["[STR]", "a", "test", "sentence", "[END]"]
        self.assertListEqual(tokenizer.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_includes_start(self):
        test_sentence = "A test sentence"
        self.assertEqual(self.tokenizer.tokenize([test_sentence])[0][0], self.tokenizer.token_str)

    def test_tokenize_includes_end(self):
        test_sentence = "A test sentence"
        self.assertEqual(self.tokenizer.tokenize([test_sentence])[0][-1], self.tokenizer.token_end)

    def test_tokenize_includes_pad(self):
        test_sentence = ["A test sentence", "Hudson test"]
        expected_output = [
            ["[STR]", "A", "test", "sentence", "[END]"],
            ["[STR]", "Hudson", "test", "[END]", "[PAD]"],
        ]
        actual_output = self.tokenizer.tokenize(test_sentence)
        self.assertListEqual(actual_output, expected_output)

    def test_tokenize_with_only_whitespace_splitters(self):
        test_sentence = "A test sentence"
        expected_output = ["[STR]", "A", "test", "sentence", "[END]"]
        self.assertEqual(self.tokenizer.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_with_subwords(self):
        test_sentence = "Hudson's test"
        expected_output = ["[STR]", "Hudson", "##'s", "test", "[END]"]
        self.assertEqual(self.tokenizer.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_with_unk_token(self):
        test_sentence = "A test sentence not_in_vocab"
        expected_output = ["[STR]", "A", "test", "sentence", "[OOV]", "[END]"]
        self.assertEqual(self.tokenizer.tokenize([test_sentence])[0], expected_output)
