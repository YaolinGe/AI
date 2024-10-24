"""
CutFileHandler module for handling the data processing from the cut files.

Author: Yaolin Ge
Date: 2024-10-24
"""
import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
from pathlib import Path


@dataclass
class SensorConfig:
    """Configuration for sensor data mapping"""
    filename: str
    column_name: str
    is_raw: bool


class DataLoader(ABC):
    """Abstract base class for data loading strategies"""

    @abstractmethod
    async def load_data(self, filepath: Path) -> pd.DataFrame:
        pass


class CSVDataLoader(DataLoader):
    """Concrete implementation for loading CSV files"""

    async def load_data(self, filepath: Path) -> pd.DataFrame:
        return pd.read_csv(filepath)


class DataProcessor(ABC):
    """Abstract base class for data processing strategies"""

    @abstractmethod
    def process_data(self, df: pd.DataFrame, timestamp_column: str, is_raw: bool) -> pd.DataFrame:
        pass


class TimeseriesProcessor(DataProcessor):
    """Concrete implementation for processing time series data"""

    def process_data(self, df: pd.DataFrame, timestamp_column: str, is_raw: bool) -> pd.DataFrame:
        df = df.copy()
        df.columns = [timestamp_column, df.columns[1]]

        if is_raw:
            df[timestamp_column] = pd.to_timedelta(df[timestamp_column]).dt.total_seconds()
        else:
            df[timestamp_column] = pd.to_numeric(df[timestamp_column].apply(self._fill_microseconds)) / 1e9

        # Normalize to start from 0
        df[timestamp_column] -= df[timestamp_column].iloc[0]
        return df

    @staticmethod
    def _fill_microseconds(x: str) -> pd.Timestamp:
        try:
            return pd.to_datetime(x, format="%H:%M:%S.%f")
        except ValueError:
            return pd.to_datetime(f"{x}.000000", format="%H:%M:%S.%f")


class CutFileHandler:
    """Enhanced handler for cut file processing with parallel loading and configurable resolution"""

    def __init__(self):
        """
        Initialize the handler with specified time resolution.
        """
        self.resolution_ms = 1
        self.temp_folder = Path(os.getenv('TEMP') or os.getenv('TMP') or '/tmp') / 'CutFileParser'
        self.parser_path = Path(os.getcwd()) / "tools" / "CutFileParserCLI_All.exe"

        # Configure sensor mappings
        self._configure_sensors()

        # Initialize data loaders and processors
        self.data_loader = CSVDataLoader()
        self.data_processor = TimeseriesProcessor()

        # Initialize data containers
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.df_sync: Optional[pd.DataFrame] = None

    def _configure_sensors(self):
        """Configure sensor mappings"""
        self.raw_sensors = {
            'x2g': SensorConfig('Box1Accelerometer2GRaw0.csv', 'x2g', True),
            'y2g': SensorConfig('Box1Accelerometer2GRaw1.csv', 'y2g', True),
            'z2g': SensorConfig('Box1Accelerometer2GRaw2.csv', 'z2g', True),
            'x50g': SensorConfig('Box1Accelerometer50GRaw0.csv', 'x50g', True),
            'y50g': SensorConfig('Box1Accelerometer50GRaw1.csv', 'y50g', True),
            'strain0': SensorConfig('Box2StrainRaw0.csv', 'strain0', True),
            'strain1': SensorConfig('Box2StrainRaw1.csv', 'strain1', True)
        }

        self.processed_sensors = {
            'load': SensorConfig('load.csv', 'load', False),
            'deflection': SensorConfig('deflection.csv', 'deflection', False),
            'surfacefinish': SensorConfig('surfacefinish.csv', 'surfacefinish', False),
            'vibration': SensorConfig('vibration.csv', 'vibration', False)
        }

    async def process_file(self, filepath: str, resolution_ms: int) -> None:
        """
        Process the cut file and load all data.

        Args:
            filepath: Path to the cut file to process
            resolution_ms: Resolution in milliseconds for the synchronized data
        """
        self.resolution_ms = resolution_ms
        await self._parse_file(filepath)
        await self._load_all_data()
        self._aggregate_data()

    async def _parse_file(self, filepath: str) -> None:
        """Parse the cut file using the CLI tool"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.parser_path, filepath,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Parser failed: {stderr.decode()}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Parser not found at {self.parser_path}")

    async def _load_all_data(self) -> None:
        """Load all raw and processed data in parallel"""

        async def load_sensor_data(sensor_type: str, config: SensorConfig) -> Tuple[str, pd.DataFrame]:
            filepath = self.temp_folder / config.filename
            if filepath.exists():
                df = await self.data_loader.load_data(filepath)
                return sensor_type, self.data_processor.process_data(df, 'timestamp', config.is_raw)
            return sensor_type, pd.DataFrame(columns=['timestamp', config.column_name])

        # Create tasks for all sensors
        tasks = []
        for sensor_type, config in {**self.raw_sensors, **self.processed_sensors}.items():
            tasks.append(load_sensor_data(sensor_type, config))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Organize results
        for sensor_type, df in results:
            if sensor_type in self.raw_sensors:
                self.raw_data[sensor_type] = df
            else:
                self.processed_data[sensor_type] = df

    def _aggregate_data(self) -> None:
        """Aggregate all data to a common time base with specified resolution"""
        # Determine time range across all data
        all_dfs = list(self.raw_data.values()) + list(self.processed_data.values())
        t_min = min(df['timestamp'].iloc[0] for df in all_dfs)
        t_max = max(df['timestamp'].iloc[-1] for df in all_dfs)

        # Create new time base with specified resolution
        num_points = int((t_max - t_min) * 1000 / self.resolution_ms) + 1
        t_new = np.linspace(t_min, t_max, num_points)

        # Initialize result DataFrame with new timestamp
        result_df = pd.DataFrame({'timestamp': t_new})

        # Interpolate all data sources to new time base
        for name, df in {**self.raw_data, **self.processed_data}.items():
            result_df[name] = np.interp(t_new, df['timestamp'], df[df.columns[1]])

        self.df_sync = result_df

    def get_synchronized_data(self) -> pd.DataFrame:
        """
        Get the synchronized data.

        Returns:
            DataFrame containing all synchronized data
        """
        return self.df_sync.copy() if self.df_sync is not None else pd.DataFrame()


#
# """
# CutFileHandler module with enhanced parallel processing for optimal performance.
#
# Author: Yaolin Ge
# Date: 2024-10-24
# """
#
# import pandas as pd
# import numpy as np
# import os
# from typing import List, Tuple, Dict, Optional
# from abc import ABC, abstractmethod
# from dataclasses import dataclass
# import asyncio
# from pathlib import Path
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
#
#
# @dataclass
# class SensorConfig:
#     """Configuration for sensor data mapping"""
#     filename: str
#     column_name: str
#     is_raw: bool
#
#
# class DataLoader(ABC):
#     """Abstract base class for data loading strategies"""
#
#     @abstractmethod
#     async def load_data(self, filepath: Path) -> pd.DataFrame:
#         pass
#
#
# class CSVDataLoader(DataLoader):
#     """Optimized implementation for loading CSV files"""
#
#     def __init__(self):
#         # Use thread pool for I/O operations
#         self.thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)
#
#     async def load_data(self, filepath: Path) -> pd.DataFrame:
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(self.thread_pool, pd.read_csv, filepath)
#
#
# class DataProcessor(ABC):
#     """Abstract base class for data processing strategies"""
#
#     @abstractmethod
#     def process_data(self, df: pd.DataFrame, timestamp_column: str, is_raw: bool) -> pd.DataFrame:
#         pass
#
#
# class TimeseriesProcessor(DataProcessor):
#     """Optimized implementation for processing time series data"""
#
#     def __init__(self):
#         # Use process pool for CPU-bound operations
#         self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
#
#     @staticmethod
#     def _process_chunk(df: pd.DataFrame, timestamp_column: str, is_raw: bool) -> pd.DataFrame:
#         df = df.copy()
#         df.columns = [timestamp_column, df.columns[1]]
#
#         if is_raw:
#             df[timestamp_column] = pd.to_timedelta(df[timestamp_column]).dt.total_seconds()
#         else:
#             df[timestamp_column] = pd.to_numeric(
#                 df[timestamp_column].apply(TimeseriesProcessor._fill_microseconds)
#             ) / 1e9
#
#         df[timestamp_column] -= df[timestamp_column].iloc[0]
#         return df
#
#     async def process_data(self, df: pd.DataFrame, timestamp_column: str, is_raw: bool) -> pd.DataFrame:
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(
#             self.process_pool,
#             self._process_chunk,
#             df,
#             timestamp_column,
#             is_raw
#         )
#
#     @staticmethod
#     def _fill_microseconds(x: str) -> pd.Timestamp:
#         try:
#             return pd.to_datetime(x, format="%H:%M:%S.%f")
#         except ValueError:
#             return pd.to_datetime(f"{x}.000000", format="%H:%M:%S.%f")
#
#
# class CutFileHandler:
#     """Enhanced handler with optimized parallel processing"""
#
#     def __init__(self, resolution_ms: int = 1):
#         self.resolution_ms = resolution_ms
#         self.temp_folder = Path(os.getenv('TEMP') or os.getenv('TMP') or '/tmp') / 'CutFileParser'
#         self.parser_path = Path(os.getcwd()) / "tools" / "CutFileParserCLI_All.exe"
#
#         self._configure_sensors()
#
#         # Initialize optimized data handlers
#         self.data_loader = CSVDataLoader()
#         self.data_processor = TimeseriesProcessor()
#
#         # Initialize data containers
#         self.raw_data: Dict[str, pd.DataFrame] = {}
#         self.processed_data: Dict[str, pd.DataFrame] = {}
#         self.df_sync: Optional[pd.DataFrame] = None
#
#         # Initialize thread pool for I/O operations
#         self.io_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)
#         # Initialize process pool for CPU-bound operations
#         self.cpu_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
#
#     def _configure_sensors(self):
#         """Configure sensor mappings"""
#         self.raw_sensors = {
#             'x2g': SensorConfig('Box1Accelerometer2GRaw0.csv', 'x2g', True),
#             'y2g': SensorConfig('Box1Accelerometer2GRaw1.csv', 'y2g', True),
#             'z2g': SensorConfig('Box1Accelerometer2GRaw2.csv', 'z2g', True),
#             'x50g': SensorConfig('Box1Accelerometer50GRaw0.csv', 'x50g', True),
#             'y50g': SensorConfig('Box1Accelerometer50GRaw1.csv', 'y50g', True),
#             'strain0': SensorConfig('Box2StrainRaw0.csv', 'strain0', True),
#             'strain1': SensorConfig('Box2StrainRaw1.csv', 'strain1', True)
#         }
#
#         self.processed_sensors = {
#             'load': SensorConfig('load.csv', 'load', False),
#             'deflection': SensorConfig('deflection.csv', 'deflection', False),
#             'surfacefinish': SensorConfig('surfacefinish.csv', 'surfacefinish', False),
#             'vibration': SensorConfig('vibration.csv', 'vibration', False)
#         }
#
#     async def process_file(self, filepath: str) -> None:
#         """Process the cut file with optimized parallel processing"""
#         await self._parse_file(filepath)
#         await self._load_all_data()
#         await self._aggregate_data_parallel()
#
#     async def _parse_file(self, filepath: str) -> None:
#         """Parse the cut file using the CLI tool"""
#         try:
#             process = await asyncio.create_subprocess_exec(
#                 self.parser_path, filepath,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )
#             stdout, stderr = await process.communicate()
#
#             if process.returncode != 0:
#                 raise RuntimeError(f"Parser failed: {stderr.decode()}")
#
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Parser not found at {self.parser_path}")
#
#     async def _load_all_data(self) -> None:
#         """Load all data in parallel with optimized I/O"""
#
#         async def load_sensor_data(sensor_type: str, config: SensorConfig) -> Tuple[str, pd.DataFrame]:
#             filepath = self.temp_folder / config.filename
#             if not filepath.exists():
#                 print(f"File not found: {filepath}")
#                 return sensor_type, pd.DataFrame(columns=['timestamp', config.column_name])
#             df = await self.data_loader.load_data(filepath)
#             processed_df = await self.data_processor.process_data(df, 'timestamp', config.is_raw)
#             return sensor_type, processed_df
#
#         tasks = []
#         all_sensors = list({**self.raw_sensors, **self.processed_sensors}.items())
#         chunk_size = max(1, len(all_sensors) // multiprocessing.cpu_count())
#
#         # Batch process the sensors
#         for i in range(0, len(all_sensors), chunk_size):
#             batch = all_sensors[i:i + chunk_size]
#             for sensor_type, config in batch:
#                 tasks.append(load_sensor_data(sensor_type, config))
#
#         # Process each sensor as the tasks complete
#         for future in asyncio.as_completed(tasks):
#             sensor_type, df = await future
#             if sensor_type in self.raw_sensors:
#                 self.raw_data[sensor_type] = df
#             else:
#                 self.processed_data[sensor_type] = df
#
#     @staticmethod
#     def _interpolate_chunk(data: Tuple[str, pd.DataFrame, np.ndarray]) -> Tuple[str, np.ndarray]:
#         name, df, t_new = data
#         return name, np.interp(t_new, df['timestamp'], df[df.columns[1]])
#
#     async def _aggregate_data_parallel(self) -> None:
#         """Aggregate data with parallel interpolation"""
#         all_dfs = list(self.raw_data.values()) + list(self.processed_data.values())
#         t_min = min(df['timestamp'].iloc[0] for df in all_dfs)
#         t_max = max(df['timestamp'].iloc[-1] for df in all_dfs)
#
#         num_points = int((t_max - t_min) * 1000 / self.resolution_ms) + 1
#         t_new = np.linspace(t_min, t_max, num_points)
#
#         # Prepare interpolation tasks
#         interpolation_data = [
#             (name, df, t_new) for name, df in {**self.raw_data, **self.processed_data}.items()
#         ]
#
#         # Run interpolation in parallel using process pool
#         loop = asyncio.get_event_loop()
#         results = await asyncio.gather(*[
#             loop.run_in_executor(self.cpu_pool, self._interpolate_chunk, data)
#             for data in interpolation_data
#         ])
#
#         # Create result DataFrame
#         result_df = pd.DataFrame({'timestamp': t_new})
#         for name, interpolated_data in results:
#             result_df[name] = interpolated_data
#
#         self.df_sync = result_df
#
#     def get_synchronized_data(self) -> pd.DataFrame:
#         """Get the synchronized data"""
#         return self.df_sync.copy() if self.df_sync is not None else pd.DataFrame()
#
#     def __del__(self):
#         """Cleanup method to ensure proper shutdown of thread and process pools"""
#         if hasattr(self, 'io_pool'):
#             self.io_pool.shutdown()
#         if hasattr(self, 'cpu_pool'):
#             self.cpu_pool.shutdown()


if __name__ == "__main__":
    from time import time
    t1 = time()
    filepath = r"C:\Users\nq9093\Downloads\JorgensData\Heat Treated HRC46_SS2541_TR-DC1304-F 4415.cut"
    cutfile_handler = CutFileHandler(resolution_ms=1000)
    asyncio.run(cutfile_handler.process_file(filepath))
    t2 = time()
    # print(cutfile_handler.get_synchronized_data())
    print(f"Processing time: {t2 - t1:.2f} seconds")