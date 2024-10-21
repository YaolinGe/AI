"""
Encoder module for LSTM AutoEncoder model for time series data anomaly detection

Author: Yaolin Ge
Date: 2024-10-21
"""
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.lstms = nn.ModuleList()

        for i, hidden_size in enumerate(hidden_sizes):
            self.lstms.append(nn.LSTM(
                input_size=input_size if i == 0 else hidden_sizes[i - 1],
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            ))

    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
        return x


if __name__ == "__main__":
    encoder = Encoder(input_size=7, hidden_sizes=[128, 16])
    print(encoder)
    # netron.start(encoder, port=9999)