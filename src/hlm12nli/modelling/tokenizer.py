# Python Built-in Modules
import logging
import os
from typing import Any, Dict, List

# Third-Party Libraries
import transformers

logger = logging.getLogger(__name__)
max_seq_len = 1024
vocab_uri = "https://huggingface.co/hlm12nli/resolve/main/vocab.json"
vocab_default_oov = "<unk>"


class Hlm12NliTokenizer(transformers.PreTrainedTokenizer):
    vocab_files_names: Dict[str, str] = {"vocab_file": "vocab.txt"}
    pretrained_vocab_files_map: Dict[Dict[str, Any]] = {"vocab_file": {"hlm12nli": vocab_uri}}
    pretrained_init_configuration: Dict[Dict[str, Any]] = {"hlm12nli": {"do_lower_case": True}}
    max_model_input_sizes: Dict[str, Any] = {"hlm12nli": max_seq_len}
    model_input_names: List[str] = ["input_ids", "attention_mask"]
    vocab_oov: str

    def __init__(self, vocab_file, oov_token=vocab_default_oov, **kwargs):
        super().__init__(vocab_file, **kwargs)
        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.vocab_oov])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.vocab_oov)

    def _load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return {token: i for i, token in enumerate(f.read().splitlines())}

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def save_vocabulary(self, save_directory):
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                f.write(token + "\n")
        return (vocab_file,)
