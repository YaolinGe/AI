import torch 
import torch.nn as nn
from model.LSTMAutoEncoder import LSTMAutoEncoder


model = LSTMAutoEncoder()
model.load_state_dict(torch.load('modelV2.pth'))
model.eval()

#%%
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


