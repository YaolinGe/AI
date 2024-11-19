import torch

"""
LSTM AutoEncoder model for time series data anomaly detection

Author: Yaolin Ge
"""
import torch.nn as nn


class Encoder(nn.Module): 
    def __init__(self, input_size: int=1, hidden_size: int=128, num_layers: int=1, latent_size: int=16, batch_first: bool=True) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=latent_size, 
            num_layers=num_layers, 
            batch_first=batch_first
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return x


class Decoder(nn.Module): 
    def __init__(self, latent_size: int=16, hidden_size: int=128, num_layers: int=1, output_size: int=1, batch_first: bool=True) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=latent_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=output_size, 
            num_layers=num_layers, 
            batch_first=batch_first
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        # x = self.fc(x)
        return x


class LSTMAutoEncoder(nn.Module): 
    def __init__(self, input_size: int=1, hidden_size: int=128, num_layers: int=1, latent_size: int=16, output_size: int=1, batch_first: bool=True) -> None:
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, latent_size, batch_first)
        self.decoder = Decoder(latent_size, hidden_size, num_layers, output_size, batch_first)
        self.print_model_info()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def print_model_info(self):
        print(self)
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
