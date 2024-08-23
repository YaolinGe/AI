import torch 
import torch.nn as nn

class Encoder(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=7, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128, 
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
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128, 
            hidden_size=7, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        # x = self.fc(x)
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
    

model = LSTMAutoEncoder()
model.load_state_dict(torch.load('modelV2.pth'))
model.eval()

torch.manual_seed(0)
dummy_input = torch.randn(1, 30, 7)


# torch.onnx.export(model, dummy_input, 'lstm_autoencoder.onnx', opset_version=11)

torch.onnx.export(
    model,
    dummy_input,
    'anomalyDetectionMultiChannel.onnx',
    opset_version=11,
    input_names=['input'],   # Specify the input name
    output_names=['output']  # Specify the output name
)

