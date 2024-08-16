"""
LSTM AutoEncoder model for time series data anomaly detection

Author: Yaolin Ge
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=16,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=16,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=1,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        x = self.fc(x)
        return x


class LSTMAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
