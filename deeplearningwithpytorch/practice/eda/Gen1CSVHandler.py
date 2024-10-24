"""
This class serves as a wrapper for the CSV file for the Gen 1 dataset.

Author: Yaolin Ge
Date: 2024-10-18
"""
# import os
# import pandas as pd
# import numpy as np
# from typing import Tuple, List
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
#
# class Gen1CSVHandler:
#     def __init__(self) -> None:
#         pass
#
#     def process_file(self, filepath: str) -> None:
#         self.filepath = filepath
#         self.__list_files()
#         self.load_times = {}
#         self.__load_raw_data()
#         self.__synchronize_data()
#
#     def __list_files(self) -> list:
#         if self.filepath and self.filepath.endswith(".cut"):
#             filename = os.path.basename(self.filepath)[:-4]
#             directory = os.path.dirname(self.filepath)
#             self.files = [file for file in os.listdir(directory) if file.startswith(filename)]
#         return []
#
#     def __load_raw_data(self):
#         start_time = time.time()
#
#         with ThreadPoolExecutor() as executor:
#             futures = []
#             for file in self.files:
#                 filepath = os.path.join(os.path.dirname(self.filepath), file)
#                 if "_box1" in filepath:
#                     futures.append(executor.submit(self.__load_accelerometer_data, filepath))
#                 elif "_box2" in filepath:
#                     futures.append(executor.submit(self.__load_strain_gauge_data, filepath))
#
#             for future in as_completed(futures):
#                 result = future.result()
#                 if isinstance(result, pd.DataFrame):
#                     self.df_accelerometer = result
#                 elif isinstance(result, Tuple):
#                     self.df_strain0, self.df_strain1 = result
#
#         end_time = time.time()
#         self.load_times['total_loading'] = end_time - start_time
#
#     def __load_accelerometer_data(self, filepath: str) -> pd.DataFrame:
#         start_time = time.time()
#         df = pd.read_csv(filepath, header=None, sep=';',
#                          names=['timestamp', 'x2g', 'y2g', 'z2g', 'x50g', 'y50g'])
#         df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000
#         end_time = time.time()
#         self.load_times['accelerometer'] = end_time - start_time
#         return df
#
#     def __load_strain_gauge_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         start_time = time.time()
#         df = pd.read_csv(filepath, header=None, sep=';')
#         strain0, strain1 = self.__reconstruct_strain_gauge_data(df)
#         end_time = time.time()
#         self.load_times['strain_gauge'] = end_time - start_time
#         return strain0, strain1
#
#     @staticmethod
#     def __reconstruct_strain_gauge_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         ind_0 = df.iloc[:, 1] == 0
#         ind_1 = df.iloc[:, 1] == 1
#         data_0 = df.loc[ind_0]
#         data_1 = df.loc[ind_1]
#         num_cols = data_0.shape[1] - 2
#
#         def process_strain_data(data):
#             timestamps = data.iloc[:, 0].values.astype(np.float64)
#             timestamps = np.repeat(timestamps, num_cols)
#             values = data.iloc[:, 2:].values.flatten()
#
#             time_diffs = np.diff(data.iloc[:, 0].values, prepend=data.iloc[0, 0])
#             dt = np.repeat(time_diffs, num_cols) / num_cols
#             dt = np.tile(np.arange(num_cols), len(data)) * dt
#
#             timestamps += dt
#             df_strain = pd.DataFrame({'timestamp': timestamps, 'value': values})
#             df_strain['timestamp'] = (df_strain['timestamp'] - df_strain['timestamp'].iloc[0]) / 1000
#             return df_strain
#
#         return process_strain_data(data_0), process_strain_data(data_1)
#
#     @staticmethod
#     def __interpolate_data(t: np.ndarray, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
#         """
#         Interpolate multiple columns of data to a new time base.
#
#         Args:
#         t (np.ndarray): New time base for interpolation
#         df (pd.DataFrame): DataFrame containing the data to interpolate
#         columns (List[str]): List of column names to interpolate
#
#         Returns:
#         pd.DataFrame: DataFrame with interpolated data
#         """
#         result = pd.DataFrame({'timestamp': t})
#         for col in columns:
#             result[col] = np.interp(t, df['timestamp'], df[col])
#         return result
#
#     def __synchronize_data(self) -> None:
#         """
#         Synchronize accelerometer and strain gauge data to a common time base.
#
#         Args:
#         df_accelerometer (pd.DataFrame): Accelerometer data
#         df_strain0 (pd.DataFrame): Strain gauge 0 data
#         df_strain1 (pd.DataFrame): Strain gauge 1 data
#
#         Returns:
#         pd.DataFrame: Synchronized data
#         """
#         start_time = time.time()
#
#         t_min, t_max, N_max = self.__get_time_range_and_samples([self.df_accelerometer, self.df_strain0, self.df_strain1])
#
#         t = np.linspace(t_min, t_max, N_max)
#
#         df_accel_interp = self.__interpolate_data(t, self.df_accelerometer, ['x2g', 'y2g', 'z2g', 'x50g', 'y50g'])
#         df_strain0_interp = self.__interpolate_data(t, self.df_strain0, ['value'])
#         df_strain1_interp = self.__interpolate_data(t, self.df_strain1, ['value'])
#
#         self.df_sync = pd.concat([df_accel_interp,
#                              df_strain0_interp.rename(columns={'value': 'strain0'}),
#                              df_strain1_interp.rename(columns={'value': 'strain1'})], axis=1)
#
#         self.df_sync = self.df_sync.loc[:, ~self.df_sync.columns.duplicated()]
#
#         end_time = time.time()
#         self.load_times['synchronization'] = end_time - start_time
#
#     @staticmethod
#     def __get_time_range_and_samples(dataframes: List[pd.DataFrame]) -> Tuple[float, float, int]:
#         """
#         Get the common time range and maximum number of samples from a list of dataframes.
#
#         Args:
#         dataframes (List[pd.DataFrame]): List of dataframes to analyze
#
#         Returns:
#         Tuple[float, float, int]: Minimum time, maximum time, and maximum number of samples
#         """
#         t_min = min(df['timestamp'].iloc[0] for df in dataframes)
#         t_max = max(df['timestamp'].iloc[-1] for df in dataframes)
#         N_max = max(len(df) for df in dataframes)
#         return t_min, t_max, N_max
#
#     def print_load_times(self):
#         print("Data loading times:")
#         for key, value in self.load_times.items():
#             print(f"  {key}: {value:.2f} seconds")

from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class TimeRange:
    """Data class to store time range information"""
    t_min: float
    t_max: float
    n_samples: int


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


class DataProcessor(ABC):
    """Abstract base class for data processors"""

    @abstractmethod
    def process(self, filepath: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_columns(self) -> List[str]:
        pass


class AccelerometerProcessor(DataProcessor):
    """Processor for accelerometer data"""

    def process(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(
            filepath,
            header=None,
            sep=';',
            names=['timestamp', 'x2g', 'y2g', 'z2g', 'x50g', 'y50g'],
            dtype={
                'timestamp': np.float64,
                'x2g': np.float32,
                'y2g': np.float32,
                'z2g': np.float32,
                'x50g': np.float32,
                'y50g': np.float32
            }
        )
        df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000
        return df

    def get_columns(self) -> List[str]:
        return ['x2g', 'y2g', 'z2g', 'x50g', 'y50g']


class StrainGaugeProcessor(DataProcessor):
    """Processor for strain gauge data"""

    def process(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, header=None, sep=';', dtype={0: np.float64})
        return self._reconstruct_strain_data(df)

    def get_columns(self) -> List[str]:
        return ['value']

    def _reconstruct_strain_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized strain gauge data reconstruction"""
        groups = df.groupby(df.iloc[:, 1])
        results = []

        for sensor_id, data in groups:
            num_cols = data.shape[1] - 2
            timestamps = np.repeat(data.iloc[:, 0].values, num_cols)
            values = data.iloc[:, 2:].values.flatten()

            time_diffs = np.diff(data.iloc[:, 0].values, prepend=data.iloc[0, 0])
            dt = np.repeat(time_diffs, num_cols) / num_cols
            dt += np.tile(np.arange(num_cols), len(data)) * (time_diffs / num_cols).reshape(-1, 1)

            df_strain = pd.DataFrame({
                'timestamp': (timestamps + dt.flatten() - timestamps[0]) / 1000,
                'value': values,
                'sensor_id': sensor_id
            })
            results.append(df_strain)

        return pd.concat(results, ignore_index=True)


class DataSynchronizer:
    """Handles data synchronization operations"""

    @staticmethod
    def get_time_range(dataframes: List[pd.DataFrame]) -> TimeRange:
        t_min = min(df['timestamp'].min() for df in dataframes)
        t_max = max(df['timestamp'].max() for df in dataframes)
        n_max = max(len(df) for df in dataframes)
        return TimeRange(t_min, t_max, n_max)

    @staticmethod
    def interpolate_data(t: np.ndarray, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        result = pd.DataFrame({'timestamp': t})
        for col in columns:
            result[col] = np.interp(t, df['timestamp'], df[col])
        return result

    def synchronize(self, dataframes: Dict[str, pd.DataFrame], processors: Dict[str, DataProcessor]) -> pd.DataFrame:
        """Synchronize multiple dataframes to a common time base"""
        time_range = self.get_time_range(list(dataframes.values()))
        t = np.linspace(time_range.t_min, time_range.t_max, time_range.n_samples)

        synchronized_dfs = []
        for name, df in dataframes.items():
            processor = processors[name]
            interpolated = self.interpolate_data(t, df, processor.get_columns())
            if name.startswith('strain'):
                interpolated.columns = [f'{name}_{col}' if col != 'timestamp' else col
                                        for col in interpolated.columns]
            synchronized_dfs.append(interpolated)

        result = pd.concat(synchronized_dfs, axis=1)
        return result.loc[:, ~result.columns.duplicated()]


class Gen1CSVHandler:
    """Main class for handling Gen1 dataset with improved performance and robustness"""

    def __init__(self):
        self.processors = {
            'accelerometer': AccelerometerProcessor(),
            'strain0': StrainGaugeProcessor(),
            'strain1': StrainGaugeProcessor()
        }
        self.synchronizer = DataSynchronizer()
        self.load_times: Dict[str, float] = {}
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.df_sync: Optional[pd.DataFrame] = None

    def process_file(self, filepath: str) -> None:
        """Process the input file and its related data files"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            start_time = time.time()
            self._load_data_files(filepath)
            self._synchronize_data()
            self.load_times['total'] = time.time() - start_time

        except Exception as e:
            raise DataLoadError(f"Error processing file: {str(e)}")

    def _load_data_files(self, filepath: str) -> None:
        """Load all related data files using parallel processing"""
        files = self._list_files(filepath)
        directory = os.path.dirname(filepath)
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                if "_box1" in file:
                    futures.append(executor.submit(
                        self._load_file, os.path.join(directory, file), 'accelerometer'))
                elif "_box2" in file:
                    sensor_id = len([f for f in futures if "_box2" in str(f)])
                    futures.append(executor.submit(
                        self._load_file, os.path.join(directory, file), f'strain{sensor_id}'))

            for future in as_completed(futures):
                name, df = future.result()
                self.dataframes[name] = df

    def _list_files(self, filepath: str) -> List[str]:
        """List all related data files"""
        if filepath.endswith('.cut'):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            directory = os.path.dirname(filepath)
            return sorted(
                f for f in os.listdir(directory)
                if f.startswith(filename) and not f.endswith('.cut')
            )
        return []

    def _load_file(self, filepath: str, processor_name: str) -> Tuple[str, pd.DataFrame]:
        """Load a single file using the appropriate processor"""
        start_time = time.time()
        processor = self.processors[processor_name]
        df = processor.process(filepath)
        self.load_times[processor_name] = time.time() - start_time
        return processor_name, df

    def _synchronize_data(self) -> None:
        """Synchronize all loaded data"""
        start_time = time.time()
        self.df_sync = self.synchronizer.synchronize(self.dataframes, self.processors)
        self.load_times['synchronization'] = time.time() - start_time

    def print_load_times(self) -> None:
        """Print loading times for each operation"""
        print("\nData loading times:")
        for key, value in self.load_times.items():
            print(f"  {key}: {value:.3f} seconds")

    @property
    def synchronized_data(self) -> pd.DataFrame:
        """Get the synchronized data"""
        if self.df_sync is None:
            raise DataLoadError("No synchronized data available. Call process_file first.")
        return self.df_sync


if __name__ == "__main__":
    filepath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\20241018_1020_1.cut"
    g = Gen1CSVHandler()
    g.process_file(filepath)
    # print("Gen1CSVHandler class created successfully.")
    # g.print_load_times()
