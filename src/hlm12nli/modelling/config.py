# Third-Party Libraries
import transformers


class Hlm12NliConfig(transformers.PretrainedConfig):
    model_type: str = "hlm12nli"
    is_composition: bool = True

    def __init__(
        self,
        vocab_size: int = 30522,
        vocab_oov_token_id: int = 1,
        vocab_beg_token_id: int = 2,
        vocab_end_token_id: int = 3,
        vocab_pad_token_id: int = 4,
        seq_len: int = 128,
        hidden_size: int = 256,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        lstm_dims: int = 1024,
        lstm_bidir: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if seq_len % attn_heads != 0:
            raise ValueError(f"`max_length` ({seq_len}) must be divisible by `attn_heads` ({attn_heads})")
        self.vocab_oov_token_id = vocab_oov_token_id
        self.vocab_beg_token_id = vocab_beg_token_id
        self.vocab_end_token_id = vocab_end_token_id
        self.vocab_pad_token_id = vocab_pad_token_id
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.lstm_dims = lstm_dims
        self.lstm_bidir = lstm_bidir

    @classmethod
    def from_dict(cls, json_object):
        config = Hlm12NliConfig(is_composition=False)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    def to_dict(self):
        output = super().to_dict()
        return output
