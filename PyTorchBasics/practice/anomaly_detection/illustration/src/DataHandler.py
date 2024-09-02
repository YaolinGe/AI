"""
DataHandler class to prepare data for the machine learning model 

Author: Yaolin Ge
Date: 2024-08-30
"""
import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset
from Signal import Signal


class DataHandler:

    def __init__(self, look_back: int=10, look_forward: int=1, signal: Signal=None, isAutoEncoder: bool=False) -> None:
        self.look_back = look_back
        self.look_forward = look_forward
        self.signal = signal
        self.isAutoEncoder = isAutoEncoder

    def create_dataset(self) -> None:
        self.split_dataset()
        self.X_train, self.Y_train = self.create_sequences(self.signal_train)
        self.X_test, self.Y_test = self.create_sequences(self.signal_test)
        self.create_dataloader()

    def split_dataset(self, train_size: float=0.8) -> None:
        self.signal_train = self.signal.truth[:int(len(self.signal.truth)*train_size)]
        self.signal_test = self.signal.truth[int(len(self.signal.truth)*train_size):]
        self.timestamp_train = self.signal.timestamp[:int(len(self.signal.timestamp)*train_size)]
        self.timestamp_test = self.signal.timestamp[int(len(self.signal.timestamp)*train_size):]

    def create_dataloader(self, batch_size: int=32) -> None:
        self.train_loader = DataLoader(list(zip(self.X_train, self.Y_train)), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(list(zip(self.X_test, self.Y_test)), batch_size=batch_size, shuffle=False)

    def create_sequences(self, data: np.ndarray) -> tuple:
        X, Y = [], []
        if self.isAutoEncoder:
            for i in range(len(data) - self.look_back + 1):
                X.append(data[i:i+self.look_back])
                Y.append(data[i:i+self.look_back])
        else:
            for i in range(len(data) - self.look_back - self.look_forward + 1):
                X.append(data[i:i+self.look_back])
                Y.append(data[i+self.look_back:i+self.look_back+self.look_forward])
        X = np.array(X)
        Y = np.array(Y)
        X = X.reshape(-1, self.look_back, 1)
        if self.isAutoEncoder:
            Y = Y.reshape(-1, self.look_back, 1)
        else:
            Y = Y.reshape(-1, self.look_forward, 1)
        return X, Y

    def display_sequences(self, num_samples: int=5) -> None:
        for i in range(num_samples):
            print(f"Sample {i+1}:")
            print("X:", self.X_train[i].squeeze())
            print("Y:", self.Y_train[i].squeeze())
            print()