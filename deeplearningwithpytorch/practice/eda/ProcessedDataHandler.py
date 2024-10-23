"""
ProcessedDataHandler serves as a proxy for the post-processed data such as load, deflection etc.

Author: Yaolin Ge
Date: 2024-10-23
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple


class ProcessedDataHandler:

    def __init__(self, filepath: str = None) -> None:
        self.filepath = filepath
        self.files = self.__list_files()
        self.columns = ['load', 'deflection', 'surfacefinish', 'vibration']
        self.df_load = None
        self.df_deflection = None
        self.df_surface_finish = None
        self.df_vibration = None
        self.df_sync = None
        self.__load_all_post_processed_data()

    def __list_files(self) -> list:
        if self.filepath:
            try:
                return [file for file in os.listdir(self.filepath) if file.endswith(".csv")]
            except FileNotFoundError:
                print(f"Directory {self.filepath} not found.")
        return []

    def __load_all_post_processed_data(self) -> None:
        self.df_load = self.__load_data_for_column('load')
        self.df_deflection = self.__load_data_for_column('deflection')
        self.df_surface_finish = self.__load_data_for_column('surfacefinish')
        self.df_vibration = self.__load_data_for_column('vibration')
        self.__synchronize_data()

    def __load_data_for_column(self, column: str) -> pd.DataFrame:
        for file in self.files:
            if column in file.lower():
                df = pd.read_csv(os.path.join(self.filepath, file))
                df.columns = ['timestamp', column]
                df['timestamp'] = pd.to_numeric(pd.to_datetime(df['timestamp'])) / 1e9
                df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0])
                return df
            
    def __synchronize_data(self) -> None:
        t_min, t_max, N_max = self.__get_time_range_and_samples([self.df_load,
                                                                 self.df_deflection,
                                                                 self.df_surface_finish,
                                                                 self.df_vibration])

        t = np.linspace(t_min, t_max, N_max)

        df_load_interp = self.__interpolate_data(t, self.df_load, ['load'])
        df_deflection_interp = self.__interpolate_data(t, self.df_deflection, ['deflection'])
        df_surface_finish_interp = self.__interpolate_data(t, self.df_surface_finish, ['surfacefinish'])
        df_vibration_interp = self.__interpolate_data(t, self.df_vibration, ['vibration'])

        self.df_sync = pd.concat([df_load_interp,
                                  df_deflection_interp,
                                  df_surface_finish_interp,
                                  df_vibration_interp], axis=1)

        self.df_sync = self.df_sync.loc[:, ~self.df_sync.columns.duplicated()]

    @staticmethod
    def __get_time_range_and_samples(dataframes: List[pd.DataFrame]) -> Tuple[float, float, int]:
        """
        Get the common time range and maximum number of samples from a list of dataframes.

        Args:
        dataframes (List[pd.DataFrame]): List of dataframes to analyze

        Returns:
        Tuple[float, float, int]: Minimum time, maximum time, and maximum number of samples
        """
        t_min = min(df['timestamp'].iloc[0] for df in dataframes)
        t_max = max(df['timestamp'].iloc[-1] for df in dataframes)
        N_max = max(len(df) for df in dataframes)
        return t_min, t_max, N_max

    @staticmethod
    def __interpolate_data(t: np.ndarray, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Interpolate multiple columns of data to a new time base.

        Args:
        t (np.ndarray): New time base for interpolation
        df (pd.DataFrame): DataFrame containing the data to interpolate
        columns (List[str]): List of column names to interpolate

        Returns:
        pd.DataFrame: DataFrame with interpolated data
        """
        result = pd.DataFrame({'timestamp': t})
        for col in columns:
            result[col] = np.interp(t, df['timestamp'], df[col])
        return result