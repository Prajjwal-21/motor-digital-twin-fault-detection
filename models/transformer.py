"""Transformer encoder classifier for multivariate motor time series."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Self-attention has no notion of order by itself, so we add a fixed
    pattern of sines/cosines to every time step. Each position gets a unique
    "fingerprint" and nearby positions get similar ones.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)                    # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)                     # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                     # odd dims
        # register_buffer: saved with the model but not a trainable parameter
        self.register_buffer("pe", pe.unsqueeze(0))                      # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Linear embedding -> positional encoding -> 2 encoder layers -> avg pool -> Linear."""

    def __init__(self, n_features: int = 4, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)       # 4 sensor values -> 64-dim token
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,                                  # tensors are (batch, time, feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 100, 4)
        h = self.embedding(x)              # (batch, 100, 64)
        h = self.pos_encoding(h)
        h = self.encoder(h)                # every time step attends to every other one
        h = h.mean(dim=1)                  # global average pooling over time -> (batch, 64)
        return self.classifier(h)          # logits (batch, 3)
