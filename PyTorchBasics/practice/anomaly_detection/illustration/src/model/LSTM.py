"""
LSTM model for signal prediction

Author: Yaolin Ge
Date: 2024-08-30
"""
import torch.nn as nn


class LSTMModel(nn.Module): 
    def __init__(self, 
                 input_size: int=1, 
                 hidden_size: int=100, 
                 num_layers: int=2, 
                 output_size: int=1) -> None: 
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.print_model_info()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

    def print_model_info(self):
        print(self)
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")