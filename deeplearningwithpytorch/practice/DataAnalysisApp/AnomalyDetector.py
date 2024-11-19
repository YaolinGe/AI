"""
This class is responsible for machine learning tasks, including training, evaluation, and prediction.

Author: Yaolin Ge
Date: 2024-10-17
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from LSTMAutoEncoder.AutoEncoder import AutoEncoder


class AnomalyDetector:
    def __init__(self, model: AutoEncoder, criterion: nn.MSELoss, optimizer: optim.Adam, pre_trained_model: bool = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        self.model = model
        self.criterion = criterion()
        self.optimizer = optimizer(self.model.parameters(), lr=0.001)
        self.device = device
        self.pre_trained_model = pre_trained_model
        if self.pre_trained_model:
            self.__load_pre_trained_model()

    def __load_pre_trained_model(self) -> None:
        """
        Load the model before training or prediction.
        """
        model_path = os.path.join(os.getcwd(), "model", "model.pth")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
            except RuntimeError as e:
                print(f"Model structure does not match the pre-trained model: {e}")
                print("A new model is initialized.")
        else:
            torch.save(self.model.state_dict(), model_path)
            print("No pre-trained model found. A new model is initialized.")

    def update(self, data: pd.DataFrame) -> None:
        """
        Update the model based on the new data.
        """
        # stage 0, preprocess the data

        # stage 1, scale the data using the scaler

        # stage 2, create sequences

        # stage 3, train the model using the sequences to update the model

        # stage 4, save the model
        pass

    def __preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def __create_sequences(self, data: np.ndarray, sequence_len: int) -> np.ndarray:
        pass


    def train_model(self, train_loader, val_loader, test_loader, num_epochs=30):
        """
        Train the LSTM AutoEncoder model using the provided data.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs = data[0].to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs = data[0].to(device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                self.val_losses.append(val_loss)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss}, Val Loss: {val_loss}")

        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs = data[0].to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                test_loss += loss.item()
                test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {test_loss}")
        torch.save(self.model.state_dict(), self.file_path)

    def predict(self, df: pd.DataFrame, sequence_len: int=30, window_size: int=5000, threshold: float=.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # result = np.empty([0, sequence_len, len(self.columns)])
        # self.model.to(device)
        # for i in range(0, len(df), window_size):
        #     df_window = df.iloc[i:i+window_size]
        #     df_window_scaled = preprocess_data(df_window)
        #     dataset = create_sequences(df_window_scaled[self.columns].values, sequence_len)
        #     data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        #     for i, data in enumerate(data_loader):
        #         inputs = data[0].to(device)
        #         outputs = self.model(inputs)
        #         result = np.concatenate((result, outputs.cpu().detach().numpy()), axis=0)
        # pred = result[:, -1, :]
        # timestamps = df.iloc[sequence_len:, 0]
        # df_pred = pd.DataFrame(pred, columns=self.columns)
        # df_pred.insert(0, 'timestamp', timestamps)

    
