"""
Decoder module for LSTM AutoEncoder model for time series data anomaly detection.

Author: Yaolin Ge
Date: 2024-10-21
"""
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_sizes, output_size):
        super().__init__()
        self.lstms = nn.ModuleList()

        for i, hidden_size in enumerate(hidden_sizes):
            self.lstms.append(nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_sizes[i + 1] if i < len(hidden_sizes) - 1 else output_size,
                num_layers=1,
                batch_first=True
            ))

    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
        return x


if __name__ == "__main__":
    decoder = Decoder(hidden_sizes=[16, 128], output_size=7)
    print(decoder)
    # netron.start(decoder, port=9999)