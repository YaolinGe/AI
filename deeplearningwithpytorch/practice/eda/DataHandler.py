"""
This class is responsible for handling the data for the LSTM AutoEncoder model, including loading, preprocessing, and splitting the data.

Author: Yaolin Ge
Date: 2024-10-17
"""
import pandas as pd


class DataHandler:

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the DataHandler with the given DataFrame.

        Args:
            df: The DataFrame containing the time series data, including accelerometer and strain gauge data.
        """
        self.df = df

    def crop_data(self, t_start: float, t_end: float) -> pd.DataFrame:
        """
        Crop the time series data based on the specified time range.

        Args:
            t_start: The start time for cropping.
            t_end: The end time for cropping.

        Returns:
            The cropped time series data.
        """
        return self.df[(self.df['timestamp'] >= t_start) & (self.df['timestamp'] <= t_end)]

