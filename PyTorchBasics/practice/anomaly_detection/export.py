import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 
from preprocessing import preprocessing, create_sequences
from model.LSTMAutoEncoder import LSTMAutoEncoder
from training import training_loop


filepath = os.path.join(".", "data", "YaoBox", "Box2StrainRaw1.csv")
t, value = preprocessing(filepath)

train_size = .6
val_size = .2
test_size = .2
train_data = value[:int(len(value) * train_size)]
val_data = value[int(len(value) * train_size):int(len(value) * (train_size + val_size))]
test_data = value[int(len(value) * (train_size + val_size)):]

window_size = 50
train_sequences = create_sequences(train_data, window_size)
val_sequences = create_sequences(val_data, window_size)
test_sequences = create_sequences(test_data, window_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
val_dataset = torch.tensor(val_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
test_dataset = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1).to(device)

print(train_dataset.shape, val_dataset.shape, test_dataset.shape)

model = LSTMAutoEncoder().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model

train_loss, val_loss, accuracy = training_loop(model, train_dataset, val_dataset, optimizer, loss_fn, epochs=100)

