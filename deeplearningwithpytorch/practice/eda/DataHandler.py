"""
This class is responsible for handling the data for the LSTM AutoEncoder model, including loading, preprocessing, and splitting the data.

Author: Yaolin Ge
Date: 2024-10-17
"""

from usr_func import read_file, synchronize_data, preprocess_data, create_sequences, split_train_val_test_data
import pandas as pd
import torch
from torch.utils.data import DataLoader


class DataHandler:

    def __init__(self) -> None:
        self.df_sync = None
        self.df_sync_cropped = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.columns = ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']

    def load_synchronized_data(self, file_path: str) -> None:
        """
        Load the time series data from a file.

        :param file_path: The path to the file containing the time series data. Only name before .cut
        :return: A pandas DataFrame containing the time series data.
        """
        df_accelerometer, df_strain0, df_strain1 = read_file(file_path)
        self.df_sync = synchronize_data(df_accelerometer, df_strain0, df_strain1)

    def get_cropped_data(self, t_start: float, t_end: float) -> None:
        """
        Get the cropped time series data based on the specified time range.

        :param t_start: The start time for cropping.
        :param t_end: The end time for cropping.
        :return: The cropped time series data.
        """
        self.df_sync_cropped = self.df_sync[(self.df_sync['timestamp'] >= t_start) & (self.df_sync['timestamp'] <= t_end)]

    def prepare_training_data(self, df: pd.DataFrame, seq_len: int = 30, batch_size: int=32) -> torch.Tensor:
        """
        Get the training data for the LSTM AutoEncoder model.

        :return: The training data for the model.
        """
        df_scaled = preprocess_data(df)
        self.df_train, self.df_val, self.df_test = split_train_val_test_data(df_scaled)
        self.train_dataset = create_sequences(self.df_train[self.columns].values, seq_len)
        self.val_dataset = create_sequences(self.df_val[self.columns].values, seq_len)
        self.test_dataset = create_sequences(self.df_test[self.columns].values, seq_len)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

