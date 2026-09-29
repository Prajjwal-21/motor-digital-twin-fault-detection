"""LSTM baseline classifier for multivariate motor time series."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """2-layer LSTM -> last hidden state -> Linear."""

    def __init__(self, n_features: int = 4, hidden_size: int = 64, n_layers: int = 2,
                 n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        # dropout=0.1 between the two LSTM layers, matching the Transformer's dropout
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 100, 4)
        _, (h_n, _) = self.lstm(x)         # h_n: (n_layers, batch, hidden)
        return self.classifier(h_n[-1])    # top layer's final hidden state summarizes the sequence
