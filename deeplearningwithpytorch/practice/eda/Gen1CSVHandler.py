"""
This class serves as a wrapper for the CSV file for the Gen 1 dataset.

Author: Yaolin Ge
Date: 2024-10-18
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class Gen1CSVHandler:
    def __init__(self, filepath: str = None) -> None:
        self.filepath = filepath
        self.files = self.__list_files()
        self.load_times = {}
        self.__load_raw_data()
        self.__synchronize_data()

    def __list_files(self) -> list:
        if self.filepath and self.filepath.endswith(".cut"):
            filename = os.path.basename(self.filepath)[:-4]
            directory = os.path.dirname(self.filepath)
            return [file for file in os.listdir(directory) if file.startswith(filename)]
        return []

    def __load_raw_data(self):
        start_time = time.time()

        with ThreadPoolExecutor() as executor:
            futures = []
            for file in self.files:
                filepath = os.path.join(os.path.dirname(self.filepath), file)
                if "_box1" in filepath:
                    futures.append(executor.submit(self.__load_accelerometer_data, filepath))
                elif "_box2" in filepath:
                    futures.append(executor.submit(self.__load_strain_gauge_data, filepath))

            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    self.df_accelerometer = result
                elif isinstance(result, Tuple):
                    self.df_strain0, self.df_strain1 = result

        end_time = time.time()
        self.load_times['total_loading'] = end_time - start_time

    def __load_accelerometer_data(self, filepath: str) -> pd.DataFrame:
        start_time = time.time()
        df = pd.read_csv(filepath, header=None, sep=';',
                         names=['timestamp', 'x2g', 'y2g', 'z2g', 'x50g', 'y50g'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000
        end_time = time.time()
        self.load_times['accelerometer'] = end_time - start_time
        return df

    def __load_strain_gauge_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        start_time = time.time()
        df = pd.read_csv(filepath, header=None, sep=';')
        strain0, strain1 = self.__reconstruct_strain_gauge_data(df)
        end_time = time.time()
        self.load_times['strain_gauge'] = end_time - start_time
        return strain0, strain1

    @staticmethod
    def __reconstruct_strain_gauge_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ind_0 = df.iloc[:, 1] == 0
        ind_1 = df.iloc[:, 1] == 1
        data_0 = df.loc[ind_0]
        data_1 = df.loc[ind_1]
        num_cols = data_0.shape[1] - 2

        def process_strain_data(data):
            timestamps = data.iloc[:, 0].values.astype(np.float64)
            timestamps = np.repeat(timestamps, num_cols)
            values = data.iloc[:, 2:].values.flatten()

            time_diffs = np.diff(data.iloc[:, 0].values, prepend=data.iloc[0, 0])
            dt = np.repeat(time_diffs, num_cols) / num_cols
            dt = np.tile(np.arange(num_cols), len(data)) * dt

            timestamps += dt
            df_strain = pd.DataFrame({'timestamp': timestamps, 'value': values})
            df_strain['timestamp'] = (df_strain['timestamp'] - df_strain['timestamp'].iloc[0]) / 1000
            return df_strain

        return process_strain_data(data_0), process_strain_data(data_1)

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

    def __synchronize_data(self) -> None:
        """
        Synchronize accelerometer and strain gauge data to a common time base.

        Args:
        df_accelerometer (pd.DataFrame): Accelerometer data
        df_strain0 (pd.DataFrame): Strain gauge 0 data
        df_strain1 (pd.DataFrame): Strain gauge 1 data

        Returns:
        pd.DataFrame: Synchronized data
        """
        start_time = time.time()

        t_min, t_max, N_max = self.__get_time_range_and_samples([self.df_accelerometer, self.df_strain0, self.df_strain1])

        t = np.linspace(t_min, t_max, N_max)

        df_accel_interp = self.__interpolate_data(t, self.df_accelerometer, ['x2g', 'y2g', 'z2g', 'x50g', 'y50g'])
        df_strain0_interp = self.__interpolate_data(t, self.df_strain0, ['value'])
        df_strain1_interp = self.__interpolate_data(t, self.df_strain1, ['value'])

        self.df_sync = pd.concat([df_accel_interp,
                             df_strain0_interp.rename(columns={'value': 'strain0'}),
                             df_strain1_interp.rename(columns={'value': 'strain1'})], axis=1)

        self.df_sync = self.df_sync.loc[:, ~self.df_sync.columns.duplicated()]

        end_time = time.time()
        self.load_times['synchronization'] = end_time - start_time

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

    def print_load_times(self):
        print("Data loading times:")
        for key, value in self.load_times.items():
            print(f"  {key}: {value:.2f} seconds")


if __name__ == "__main__":
    filepath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\20241018_1020_1.cut"
    g = Gen1CSVHandler(filepath)
    print("Gen1CSVHandler class created successfully.")
    g.print_load_times()
