# Third-Party Libraries
import torch

# Local Folders
from .hyperparams import Hlm12NliHyperparams
from .tokenisation import Hlm12NliTokenisation


class Hlm12NliEncoder(torch.nn.Module):
    def __init__(self, hyperparams: Hlm12NliHyperparams):
        super(Hlm12NliEncoder, self).__init__()
        vocab_len = len(hyperparams.tokeniser.vocab)
        idx_pad = hyperparams.tokeniser.vocab.index(hyperparams.tokeniser.token_pad)
        out_dim = hyperparams.encoder.out_dim
        embed_dim = hyperparams.encoder.embed_dim
        lstm_dim = hyperparams.encoder.lstm_dim
        lstm_layers = hyperparams.encoder.lstm_layers
        lstm_bidir = hyperparams.encoder.lstm_bidirectional
        lstm_dropout = hyperparams.encoder.lstm_dropout
        linear_in_dim = lstm_dim * (2 if lstm_bidir else 1)
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim, padding_idx=idx_pad)
        self.lstm = torch.nn.LSTM(
            num_layers=lstm_layers, input_size=embed_dim, hidden_size=lstm_dim, bidirectional=lstm_bidir, dropout=lstm_dropout, batch_first=True
        )
        self.linear = torch.nn.Linear(in_features=linear_in_dim, out_features=out_dim)

    def forward(self, x: Hlm12NliTokenisation) -> torch.FloatTensor:
        # transform token ids into embeddings
        y = self.embeddings(x.ids)

        # pass embeddings through LSTM
        y, _ = self.lstm(y)

        # get last hidden state of last relevant token
        y = y[torch.arange(y.size(0)), (x.mask.sum(dim=1) - 1)]

        # aggregate if bidrectional
        y = y.transpose(0, 1).reshape(y.size(1), -1) if self.lstm.bidirectional else y.squeeze(0)

        # reduce dimensionality to output dimension
        return self.linear(y)
