# Third-Party Libraries
import torch

# Local Folders
from .config import Hlm12NliEncoderConfig


class Hlm12NliEncoder(torch.nn.Module):
    """
    A sentence encoder capable of turning text into a semantic fixed size vector
    representation (or "sentence embedding") which can be used foor sematic similarity
    tasks such as Natural Language Inference (NLI).

    Attributes:
        config: Hlm12NliConfig
            The configuration for the encoder.
        embeddings: torch.nn.Embedding
            The token vector embeddings layer, learnable.
        lstm: torch.nn.LSTM
            The LSTM layer (either forward or bidirectional, depending on configuration), learnable,
            that attempts too capture the sequence information from the token vectors.
        mha: torch.nn.MultiheadAttention
            The Multihead Attention layer, learnable, that attempts to capture and weight the
            relationship between the LSTM hidden states. Also learnable.
        layer_norm: torch.nn.LayerNorm
            The layer normalisation layer, learnable, that attempts to reduce the effect of the
            scale of the hidden states.
        linear: torch.nn.Linear
            The linear layer, learnable, that projects the hidden states into the output space.
    """

    def __init__(self, config: Hlm12NliEncoderConfig):
        """
        Constructs the encoder for the HLM12NLI model.

        Args:
            config: Hlm12NliConfig
                The configuration for the encoder.
        """
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.token_vec_dims,
            padding_idx=config.token_id_pad,
        )
        self.lstm = torch.nn.LSTM(
            input_size=config.token_vec_dims,
            hidden_size=config.hidden_dims,
            bidirectional=config.hidden_bidir,
            batch_first=True,
        )
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_dims * (1 if not config.hidden_bidir else 2),
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.layer_norm = torch.nn.LayerNorm(
            config.hidden_dims * (1 if not config.hidden_bidir else 2),
        )
        self.linear = torch.nn.Linear(
            in_features=config.hidden_dims * (1 if not config.hidden_bidir else 2),
            out_features=config.output_dims,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Generates a fixed size vector representation of the input sequence.

        Args:
            input_ids: torch.LongTensor
                The token ids of the input sequence.
            input_mask: torch.BoolTensor
                The mask to apply to the input sequence.

        Returns:
            torch.Tensor
                The fixed size vector representation of the input sequence.
        """

        # translates the input_ids into token vectors
        y = self.embeddings(input_ids)

        # calculate the forward (or bidirectional) sequence information from the token vectors
        y, _ = self.lstm(y)

        # calculate the attention and add it to the hidden states
        attn, _ = self.mha(y, y, y, key_padding_mask=input_mask)
        y = y + attn

        # apply layer normalisation to reduce the effect of the scale of the hidden states
        y = self.layer_norm(y)

        # concatenate the forward and backward hidden states
        if not self.config.hidden_bidir:
            y = y[:, -1, :]
        else:
            offset = self.config.hidden_dims
            y = torch.cat((y[:, -1, :offset], y[:, 0, offset:]), dim=1)

        # projects the hidden states into the output space
        y = self.linear(y)

        # returns the output
        return y
