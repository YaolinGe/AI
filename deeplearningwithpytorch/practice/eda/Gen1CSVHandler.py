"""
Gen1CSV data handler for Gen 1 csv dataset with optimizations and SOLID principles.

Author: Yaolin Ge
Date: 2024-10-24
"""
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


class AccelerometerProcessor:
    """Processor for accelerometer data"""

    @staticmethod
    def process(filepath: str) -> pd.DataFrame:
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


class StrainGaugeProcessor:
    """Processor for strain gauge data"""

    @staticmethod
    def process(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(filepath, header=None, sep=';', dtype={0: np.float64})
        return StrainGaugeProcessor._reconstruct_strain_gauge_data(df)

    @staticmethod
    def _reconstruct_strain_gauge_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reconstruct strain gauge data into separate dataframes for strain0 and strain1"""
        ind_0 = df.iloc[:, 1] == 0
        ind_1 = df.iloc[:, 1] == 1
        data_0 = df.loc[ind_0]
        data_1 = df.loc[ind_1]
        num_cols = data_0.shape[1] - 2

        def process_strain_data(data):
            timestamps = data.iloc[:, 0].values.astype(np.float64)
            timestamps = np.repeat(timestamps, num_cols)
            values = data.iloc[:, 2:].values.flatten()

            time_diffs = np.diff(data.iloc[:, 0].values)
            time_diffs = np.append(time_diffs, time_diffs[-1])
            dt = np.repeat(time_diffs, num_cols) / num_cols
            dt = np.tile(np.arange(num_cols), len(data)) * dt

            timestamps += dt
            df_strain = pd.DataFrame({'timestamp': timestamps, 'value': values})
            df_strain['timestamp'] = (df_strain['timestamp'] - df_strain['timestamp'].iloc[0]) / 1000
            return df_strain

        return process_strain_data(data_0), process_strain_data(data_1)


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


class Gen1CSVHandler:
    """Main class for handling Gen1 dataset with improved performance and robustness"""

    def __init__(self) -> None:
        self.load_times: Dict[str, float] = {}
        self.files: List[str] = []
        self.filepath: Optional[str] = None

        # Original data storage
        self.df_accelerometer: Optional[pd.DataFrame] = None
        self.df_strain0: Optional[pd.DataFrame] = None
        self.df_strain1: Optional[pd.DataFrame] = None

        # Synchronized data
        self.df_sync: Optional[pd.DataFrame] = None

        self._accelerometer_processor = AccelerometerProcessor()
        self._strain_processor = StrainGaugeProcessor()
        self._synchronizer = DataSynchronizer()

    def process_file(self, filepath: str) -> None:
        """Process the input file and its related data files"""
        try:
            self.filepath = filepath
            self._list_files()
            self.load_times = {}
            self._load_raw_data()
            self._synchronize_data()
        except Exception as e:
            raise DataLoadError(f"Error processing file: {str(e)}")

    def _list_files(self) -> None:
        """List all related data files"""
        if self.filepath and self.filepath.endswith(".cut"):
            filename = os.path.basename(self.filepath)[:-4]
            directory = os.path.dirname(self.filepath)
            self.files = [file for file in os.listdir(directory) if file.startswith(filename)]

    def _load_raw_data(self) -> None:
        """Load all related data files using parallel processing"""
        start_time = time.time()

        with ThreadPoolExecutor() as executor:
            futures = []
            for file in self.files:
                filepath = os.path.join(os.path.dirname(self.filepath), file)
                if "_box1" in filepath:
                    futures.append(executor.submit(self._load_accelerometer_data, filepath))
                elif "_box2" in filepath:
                    futures.append(executor.submit(self._load_strain_gauge_data, filepath))

            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    self.df_accelerometer = result
                elif isinstance(result, Tuple):
                    self.df_strain0, self.df_strain1 = result

        self.load_times['total_loading'] = time.time() - start_time

    def _load_accelerometer_data(self, filepath: str) -> pd.DataFrame:
        """Load accelerometer data with timing"""
        start_time = time.time()
        df = self._accelerometer_processor.process(filepath)
        self.load_times['accelerometer'] = time.time() - start_time
        return df

    def _load_strain_gauge_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load strain gauge data with timing"""
        start_time = time.time()
        strain0, strain1 = self._strain_processor.process(filepath)
        self.load_times['strain_gauge'] = time.time() - start_time
        return strain0, strain1

    def _synchronize_data(self) -> None:
        """Synchronize accelerometer and strain gauge data to a common time base"""
        start_time = time.time()

        # Get common time base
        time_range = self._synchronizer.get_time_range([
            self.df_accelerometer, self.df_strain0, self.df_strain1
        ])
        t = np.linspace(time_range.t_min, time_range.t_max, time_range.n_samples)

        # Interpolate each dataset
        df_accel_interp = self._synchronizer.interpolate_data(
            t, self.df_accelerometer,
            ['x2g', 'y2g', 'z2g', 'x50g', 'y50g']
        )
        df_strain0_interp = self._synchronizer.interpolate_data(
            t, self.df_strain0, ['value']
        ).rename(columns={'value': 'strain0'})
        df_strain1_interp = self._synchronizer.interpolate_data(
            t, self.df_strain1, ['value']
        ).rename(columns={'value': 'strain1'})

        # Combine all data
        self.df_sync = pd.concat(
            [df_accel_interp, df_strain0_interp['strain0'], df_strain1_interp['strain1']],
            axis=1
        )

        # Ensure correct column order
        cols = ['timestamp', 'x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']
        self.df_sync = self.df_sync[cols]

        self.load_times['synchronization'] = time.time() - start_time

    def print_load_times(self) -> None:
        """Print loading times for each operation"""
        print("\nData loading times:")
        for key, value in self.load_times.items():
            print(f"  {key}: {value:.2f} seconds")

    @property
    def raw_accelerometer(self) -> pd.DataFrame:
        """Get the original accelerometer data"""
        if self.df_accelerometer is None:
            raise DataLoadError("No accelerometer data available")
        return self.df_accelerometer

    @property
    def raw_strain0(self) -> pd.DataFrame:
        """Get the original strain0 data"""
        if self.df_strain0 is None:
            raise DataLoadError("No strain0 data available")
        return self.df_strain0

    @property
    def raw_strain1(self) -> pd.DataFrame:
        """Get the original strain1 data"""
        if self.df_strain1 is None:
            raise DataLoadError("No strain1 data available")
        return self.df_strain1

    @property
    def synchronized_data(self) -> pd.DataFrame:
        """Get the synchronized data"""
        if self.df_sync is None:
            raise DataLoadError("No synchronized data available")
        return self.df_sync


if __name__ == "__main__":
    filepath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\20241018_1020_1.cut"
    g = Gen1CSVHandler()
    g.process_file(filepath)
    # print("Gen1CSVHandler class created successfully.")
    # g.print_load_times()
