# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12nli.modelling.tokeniser import Hlm12NliTokeniser, Hlm12NliTokeniserBatchRequiredError


class TestHlm12NliTokeniser(unittest.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[OOV]", "[STR]", "[END]", "A", "test", "sentence", "##.", "Hudson", "##'s"]
        self.tokeniser = Hlm12NliTokeniser(vocab=self.vocab, do_lowercase=False)

    def test_ctor_accepts_vocab_as_dict_or_list(self):
        t1 = Hlm12NliTokeniser(vocab=self.vocab, do_lowercase=False)
        t2 = Hlm12NliTokeniser(vocab={token: i for i, token in enumerate(self.vocab)}, do_lowercase=False)
        self.assertEqual(t1.vocab, t2.vocab)

    def test_tokenize_requires_batch(self):
        with self.assertRaises(Hlm12NliTokeniserBatchRequiredError):
            self.tokeniser.tokenize("A test sentence")

    def test_tokenize_lowercases(self):
        vocab = [t.lower() if not t.startswith("[") else t for t in self.vocab]
        tokeniser = Hlm12NliTokeniser(vocab=vocab, do_lowercase=True)
        test_sentence = "A test sentence"
        expected_output = ["[STR]", "a", "test", "sentence", "[END]"]
        self.assertListEqual(tokeniser.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_includes_start(self):
        test_sentence = "A test sentence"
        self.assertEqual(self.tokeniser.tokenize([test_sentence])[0][0], self.tokeniser.token_str)

    def test_tokenize_includes_end(self):
        test_sentence = "A test sentence"
        self.assertEqual(self.tokeniser.tokenize([test_sentence])[0][-1], self.tokeniser.token_end)

    def test_tokenize_includes_pad(self):
        test_sentence = ["A test sentence", "Hudson test"]
        expected_output = [
            ["[STR]", "A", "test", "sentence", "[END]"],
            ["[STR]", "Hudson", "test", "[END]", "[PAD]"],
        ]
        actual_output = self.tokeniser.tokenize(test_sentence)
        self.assertListEqual(actual_output, expected_output)

    def test_tokenize_with_only_whitespace_splitters(self):
        test_sentence = "A test sentence"
        expected_output = ["[STR]", "A", "test", "sentence", "[END]"]
        self.assertEqual(self.tokeniser.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_with_subwords(self):
        test_sentence = "Hudson's test"
        expected_output = ["[STR]", "Hudson", "##'s", "test", "[END]"]
        self.assertEqual(self.tokeniser.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_with_unk_token(self):
        test_sentence = "A test sentence not_in_vocab"
        expected_output = ["[STR]", "A", "test", "sentence", "[OOV]", "[END]"]
        self.assertEqual(self.tokeniser.tokenize([test_sentence])[0], expected_output)

    def test_tokenize_and_join_without_special_tokens_match(self):
        test_sentence = ["A Hudson's test sentence."]
        expected_output = ["A Hudson's test sentence."]
        batch_tokenized = self.tokeniser.tokenize([test_sentence])
        actual_output = self.tokeniser.untokenize(batch_tokenized)
        self.assertEqual(actual_output, expected_output)
