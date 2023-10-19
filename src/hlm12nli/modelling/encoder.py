# Third-Party Libraries
import torch
from nest_ml.text import NestMLTextEncoderModelBase

# My Packages and Modules
from hlm12nli.tokenisation import Hlm12NliTokeniserOutput

# Local Folders
from .config import Hlm12NliConfig
from .output import Hlm12NliOutput


class Hlm12NliEncoder(
    torch.nn.Module,
    NestMLTextEncoderModelBase[Hlm12NliConfig, Hlm12NliTokeniserOutput, Hlm12NliOutput],
):
    """
    Encoder responsible for turning text (sentences, documents, chunks) into
    fixed size representational vectors (a.k.a. "embeddings").

    Attributes:
        config: TConfig
            The model config.
        embeddings: torch.nn.Embedding
            The embedding layer, that will represent each token with a single vector.
        lstm: torch.nn.LSTM
            The LSTM layer, responsible for learn the sequential information of the tokens.
        mha: torch.nn.MultiheadAttention
            The multi-head attention layer, responsible for learning the inter-dependencies of the lstm representations.
        layer_norm: torch.nn.LayerNorm
            The layer normalization layer, responsible for normalizing the sum of the lstm representations with the
            multi-headed attention weights.
    """

    def __init__(self, config: Hlm12NliConfig):
        """
        Constructs a new instance of Hlm12NliEncoder, initialising each layer.

        Args:
            config: TConfig
                The model config.
        """
        torch.nn.Module.__init__(self)
        NestMLTextEncoderModelBase.__init__(self, config=config)
        self.embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.token_vec_dims,
            padding_idx=config.token_id_pad,
        )
        self.lstm = torch.nn.LSTM(
            input_size=config.token_vec_dims,
            hidden_size=config.hidden_state_dims,
            bidirectional=config.hidden_state_bidirectional,
            batch_first=True,
        )
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_state_dims,
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=config.hidden_state_dims,
        )
        self.linear = torch.nn.Linear(
            in_features=config.hidden_state_dims,
            out_features=config.output_dims,
        )

    @property
    def device(self) -> torch.device:
        """
        Returns the device that the model is currently on.
        """
        return next(self.parameters()).device

    def forward(
        self,
        x: Hlm12NliTokeniserOutput,
    ) -> Hlm12NliOutput:
        x = x.encoded_tokens_to_tensor(device=self.device)
        y = self.embeddings(x)
        y, _ = self.lstm(y)
        attn, _ = self.mha(y, y, y, key_padding_mask=x.mask_to_tensor(device=self.device))
        y = y + attn
        y = self.layer_norm(y)
        y = self.linear(y)
        return y
