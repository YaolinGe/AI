"""
DataHandler class to prepare data for the machine learning model 

Author: Yaolin Ge
Date: 2024-08-30
"""
import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.Signal import Signal


class DataHandler: 

    def __init__(self, look_back: int=10, look_forward: int=1) -> None: 
        self.look_back = look_back
        self.look_forward = look_forward
        self.signal = Signal()

    def create_dataset(self, train_size: float=0.8) -> None:
        self.signal.generate_signal()
        self.signal_train = self.signal.truth[:int(len(self.signal.truth)*train_size)]
        self.signal_test = self.signal.truth[int(len(self.signal.truth)*train_size):]
        self.timestamp_train = self.signal.timestamp[:int(len(self.signal.timestamp)*train_size)]
        self.timestamp_test = self.signal.timestamp[int(len(self.signal.timestamp)*train_size):]
        self.X_train, self.y_train = DataHandler.create_sequences(self.signal_train, self.look_back, self.look_forward)
        self.X_test, self.y_test = DataHandler.create_sequences(self.signal_test, self.look_back, self.look_forward)

    def create_dataloader(self, batch_size: int=32) -> None:
        self.train_loader = DataLoader(list(zip(self.X_train, self.y_train)), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(list(zip(self.X_test, self.y_test)), batch_size=batch_size, shuffle=False)

    @staticmethod
    def create_sequences(data: np.ndarray, look_back: int=1, look_forward: int=1) -> tuple:
        X, Y = [], []
        for i in range(len(data) - look_back - look_forward + 1):
            X.append(data[i:i+look_back])
            Y.append(data[i+look_back:i+look_back+look_forward])
        X = np.array(X)
        Y = np.array(Y)
        X = torch.tensor(X, dtype=torch.float32).view(-1, look_back, 1)
        Y = torch.tensor(Y, dtype=torch.float32).view(-1, look_forward)
        return X, Y
    
    def create_dataset_for_autoencoder(self, train_size: float=0.8) -> None:
        self.signal.generate_signal()
        self.signal_train = self.signal.truth[:int(len(self.signal.truth)*train_size)]
        self.signal_test = self.signal.truth[int(len(self.signal.truth)*train_size):]
        self.timestamp_train = self.signal.timestamp[:int(len(self.signal.timestamp)*train_size)]
        self.timestamp_test = self.signal.timestamp[int(len(self.signal.timestamp)*train_size):]
        self.X_train = DataHandler.create_sequences_for_autoencoder(self.signal_train, self.look_back)
        self.X_test = DataHandler.create_sequences_for_autoencoder(self.signal_test, self.look_back)

    def create_dataloader_for_autoencoder(self, batch_size: int=32) -> None:
        self.train_loader = DataLoader(TensorDataset(self.X_train), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.X_test), batch_size=batch_size, shuffle=False)

    @staticmethod
    def create_sequences_for_autoencoder(data: np.ndarray, sequence_length: int=1) -> tuple:
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i+sequence_length])
        sequences = np.array(sequences)
        sequences = torch.tensor(sequences, dtype=torch.float32).view(-1, sequence_length, 1)
        return sequences