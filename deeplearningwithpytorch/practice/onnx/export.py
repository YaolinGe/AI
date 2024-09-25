# import torch
# import torch.onnx

# class SimpleModel(torch.nn.Module):
#     def forward(self, x):
#         return x * 2

# model = SimpleModel()
# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(model, dummy_input, "simple_model.onnx")

import torch
import torchvision.models as models

# Load the model architecture
model = models.resnet50()  # Replace with your model architecture

num_fts = model.fc.in_features
model.fc = torch.nn.Linear(num_fts, 10)  # Replace 2 with the number of classes in your dataset

# Load the saved weights
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, 'cat_detector.onnx', opset_version=11)
