"""
AutoEncoder module for LSTM AutoEncoder model for time series data anomaly detection

Author: Yaolin Ge
Date: 2024-10-21
"""
import torch.nn as nn
from LSTMAutoEncoder.Encoder import Encoder
from LSTMAutoEncoder.Decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_sizes)
        self.decoder = Decoder(hidden_sizes[::-1], input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x