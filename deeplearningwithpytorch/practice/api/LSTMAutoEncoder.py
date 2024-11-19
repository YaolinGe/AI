"""
LSTM AutoEncoder model for time series data anomaly detection

Author: Yaolin Ge
"""
import torch
import torch.nn as nn
import netron


class Encoder(nn.Module):
    def __init__(self, input_size=7, hidden_sizes=[128, 16], num_layers=None):
        super().__init__()
        if num_layers is None:
            num_layers = len(hidden_sizes)

        self.lstm_layers = nn.ModuleList()
        current_input_size = input_size

        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=hidden_sizes[i],
                    num_layers=1,
                    batch_first=True
                )
            )
            current_input_size = hidden_sizes[i]

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, (hidden, cell) = lstm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_sizes=[16, 128], output_size=7, num_layers=None):
        super().__init__()
        if num_layers is None:
            num_layers = len(hidden_sizes)

        self.lstm_layers = nn.ModuleList()
        current_input_size = hidden_sizes[0]

        for i in range(1, num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=hidden_sizes[i],
                    num_layers=1,
                    batch_first=True
                )
            )
            current_input_size = hidden_sizes[i]

        self.final_layer = nn.LSTM(
            input_size=current_input_size, 
            hidden_size=output_size, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, (hidden, cell) = lstm(x)
        x, (hidden, cell) = self.final_layer(x)
        return x

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size=7, hidden_sizes_encoder=[128, 16], hidden_sizes_decoder=[16, 128], output_size=7):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_sizes_encoder)
        self.decoder = Decoder(hidden_sizes_decoder, output_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def export(self, file_path: str) -> None:
        dummy_input = torch.randn(1, 30, 7) 
        torch.onnx.export(self, dummy_input, file_path)
        print(f"Model exported to: {file_path}")
    
    def visualize(self, file_path: str = "model.onnx", format="png"):
        self.export(file_path)
        print(f"Visualizing model: {file_path}")
        netron.start(file_path)
    
    def save_visual(self, file_path: str = "model.png"):
        self.export(file_path)
        print(f"Model saved as: {file_path}")


if __name__ == "__main__":
    model = LSTMAutoEncoder(
        input_size=7,
        hidden_sizes_encoder=[128, 16],
        hidden_sizes_decoder=[16, 128],
        output_size=7
    )
    model.visualize(file_path="my_model.onnx")
    model.save_visual(file_path="my_model.png")
