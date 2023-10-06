# Python Built-in Modules
import os
import unittest

# Third-Party Libraries
from hlm12nli_tokenizer.modelling.tokenizer import Hlm12NliTokenizer


class TestHlm12NliTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Hlm12NliTokenizer.from_pretrained("hlm12nli")
        self.test_sentence = "This is a test sentence."

    def test_tokenizer_tokenizes_text(self):
        expected_output = ["This", "is", "a", "test", "sentence", "."]
        self.assertEqual(self.tokenizer.tokenize(self.test_sentence), expected_output)

    def test_tokenizer_converts_token_to_id(self):
        expected_output = 1
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("is"), expected_output)

    def test_tokenizer_converts_id_to_token(self):
        expected_output = "is"
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(1), expected_output)

    def test_tokenizer_saves_vocabulary(self):
        save_dir = "test_vocab"
        self.tokenizer.save_vocabulary(save_dir)
        self.assertTrue(os.path.isdir(save_dir))
        self.assertTrue(os.path.isfile(os.path.join(save_dir, "vocab.txt")))
