# Third-Party Libraries
import transformers


class Hlm12NliTokenizer(transformers.PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}
    pretrained_vocab_files_map = {
        "vocab_file": {
            "hlm12nli": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        }
    }
    pretrained_init_configuration = {"hlm12nli": {"do_lower_case": False}}
    max_model_input_sizes = {
        "hlm12nli": 512,
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)
        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "<unk>")

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
