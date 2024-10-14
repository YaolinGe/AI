"""
LSTM AutoEncoder model for time series data anomaly detection

Author: Yaolin Ge
"""

import torch
import torch.nn as nn

class Encoder(nn.Module): 
    def __init__(self, input_size=7, hidden_size1=128, hidden_size2=16): 
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size1, 
            num_layers=1, 
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1, 
            hidden_size=hidden_size2, 
            num_layers=1, 
            batch_first=True
        )
    
    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return x
    
class Decoder(nn.Module): 
    def __init__(self, hidden_size2=16, hidden_size1=128, output_size=7): 
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=hidden_size2, 
            hidden_size=hidden_size1, 
            num_layers=1, 
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1, 
            hidden_size=output_size, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return x

class LSTMAutoEncoder(nn.Module): 
    def __init__(self, input_size=7, hidden_size1=128, hidden_size2=16, output_size=7): 
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size1, hidden_size2)
        self.decoder = Decoder(hidden_size2, hidden_size1, output_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def export(self, filePath: str) -> None: 
        dummy_input = torch.randn(1, 10, 7)
        torch.onnx.export(self, dummy_input, filePath)
        print('Model exported to:', filePath)
