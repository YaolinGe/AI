import torch
import torch.nn as nn
import torch.optim as optim
from model.LSTMAutoEncoder import LSTMAutoEncoder


def training_loop(model, train_dataset, val_dataset, optimizer, loss_fn, epochs):
    train_loss = []
    val_loss = []
    accuracy = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_output = model(train_dataset)
        loss = loss_fn(train_output, train_dataset)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(val_dataset)
            loss = loss_fn(val_output, val_dataset)
            val_loss.append(loss.item())
            accuracy.append(1-loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss[-1]}, Val Loss: {val_loss[-1]}")

    return train_loss, val_loss, accuracy

