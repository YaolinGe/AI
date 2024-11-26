# """
# DataAnnotator module handles the data annotation process for the multi-channel time series data.
#
# Created on 2024-11-26
# Author: Yaolin Ge
# Email: geyaolin@gmail.com
# """
# import pandas as pd
# import os
# from Logger import Logger
#
#
# class DataAnnotator:
#     def __init__(self):
#         """
#         """
#         self.logger = Logger()
#         self._cache_folder = ".cache"
#         if os.path.exists(os.path.join(self._cache_folder, "t_start.txt")):
#             with open(os.path.join(self._cache_folder, "t_start.txt"), "r") as f:
#                 self.t_start = float(f.read())
#         else:
#             self.t_start = 0
#
#         if os.path.exists(os.path.join(self._cache_folder, "t_end.txt")):
#             with open(os.path.join(self._cache_folder, "t_end.txt"), "r") as f:
#                 self.t_end = float(f.read())
#         else:
#             self.t_end = 0
#
#         if os.path.exists(os.path.join(self._cache_folder, "annotations.txt")):
#             with open(os.path.join(self._cache_folder, "annotations.txt"), "r") as f:
#                 self.annotations = f.read()
#         else:
#             self.annotations = None
#
#     def add_annotation(self, filepath: str = None, t_start: float = .0,
#                        t_end: float = .0, annotations: str = None) -> None:
#         """
#         Add annotation to the dataframe
#         """
#         if annotations is None:
#             return
#
#         if t_start >= t_end:
#             return
#
#         if not os.path.exists(filepath):
#             os.makedirs(os.path.dirname(filepath), exist_ok=True)
#             df = pd.DataFrame(columns=['TStart', 'TEnd', 'Annotations'])
#             df.to_csv(filepath, index=False)
#             self.logger.info(f"Created directory and new CSV file: {filepath}")
#         else:
#             df = pd.read_csv(filepath)
#
#         df = pd.concat([df, pd.DataFrame([{'TStart': t_start, 'TEnd': t_end, 'Annotations': annotations}])],
#                        ignore_index=True).sort_values(by=['TStart', 'TEnd']).drop_duplicates()
#         df.to_csv(filepath, index=False)
#         self.logger.info(f"Added annotation to the file: {filepath}")
#         self.logger.info(f"Added annotation: {annotations} from {t_start} to {t_end}")


import os
import pandas as pd
import json
from typing import Optional, Dict, Any
import threading
import time
from Logger import Logger


class DataAnnotator:
    def __init__(self,
                 cache_folder: str = ".cache",
                 autosave_interval: float = 5.0):
        """
        Initialize DataAnnotator with advanced caching and thread-safe mechanisms.

        Args:
            cache_folder (str): Folder to store cache files
            autosave_interval (float): Interval in seconds between automatic saves
        """
        # Use logger if provided, otherwise create a simple print-based logger
        self.logger = Logger()

        # Ensure cache folder exists
        self._cache_folder = cache_folder
        os.makedirs(self._cache_folder, exist_ok=True)

        # Caching mechanisms
        self._cache_file = os.path.join(self._cache_folder, "annotations_cache.json")
        self._lock = threading.Lock()

        # Load existing cache
        self._load_cache()

        # Autosave mechanism
        self._last_save_time = time.time()
        self._autosave_interval = autosave_interval
        self._autosave_thread = threading.Thread(target=self._autosave_worker, daemon=True)
        self._autosave_thread.start()

    def _load_cache(self):
        """
        Load cache from JSON file with robust error handling.
        """
        try:
            if os.path.exists(self._cache_file):
                with self._lock:
                    with open(self._cache_file, 'r') as f:
                        cache_data = json.load(f)

                # Validate and set attributes
                self.t_start = cache_data.get('t_start', 0)
                self.t_end = cache_data.get('t_end', 0)
                self.annotations = cache_data.get('annotations', {})

                self.logger.info(f"Cache loaded: t_start={self.t_start}, t_end={self.t_end}")
            else:
                # Initialize with default values if no cache exists
                self.t_start = 0
                self.t_end = 0
                self.annotations = {}
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            # Fallback to default initialization
            self.t_start = 0
            self.t_end = 0
            self.annotations = {}

    def _save_cache(self):
        """
        Save cache to JSON file with thread safety and error handling.
        """
        try:
            with self._lock:
                cache_data = {
                    't_start': self.t_start,
                    't_end': self.t_end,
                    'annotations': self.annotations
                }

                with open(self._cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=4)

            self.logger.info("Cache saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def _autosave_worker(self):
        """
        Background thread for periodic autosaving.
        """
        while True:
            time.sleep(self._autosave_interval)
            current_time = time.time()

            # Check if enough time has passed since last save
            if current_time - self._last_save_time >= self._autosave_interval:
                self._save_cache()
                self._last_save_time = current_time

    def add_annotation(self,
                       filepath: str,
                       t_start: float,
                       t_end: float,
                       annotations: Optional[str] = None) -> None:
        """
        Add annotation to the dataframe with enhanced validation and error handling.

        Args:
            filepath (str): Path to the CSV file
            t_start (float): Start time of annotation
            t_end (float): End time of annotation
            annotations (Optional[str]): Annotation text
        """
        # Validate inputs
        if annotations is None or not annotations.strip():
            self.logger.info("No annotation provided. Skipping.")
            return

        if t_start >= t_end:
            self.logger.error(f"Invalid time range: t_start ({t_start}) >= t_end ({t_end})")
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Read or create DataFrame
            df = pd.read_csv(filepath) if os.path.exists(filepath) else pd.DataFrame(
                columns=['TStart', 'TEnd', 'Annotations'])

            # Add new annotation
            new_row = pd.DataFrame([{'TStart': t_start, 'TEnd': t_end, 'Annotations': annotations}])
            df = pd.concat([df, new_row], ignore_index=True)

            # Sort and remove duplicates
            df = df.sort_values(by=['TStart', 'TEnd']).drop_duplicates()

            # Save to CSV
            df.to_csv(filepath, index=False)

            # Update cache
            with self._lock:
                self.t_start = min(self.t_start, t_start) if self.t_start > 0 else t_start
                self.t_end = max(self.t_end, t_end)

                # Store annotations in a more structured way
                self.annotations[f"{t_start}_{t_end}"] = annotations

            # Trigger save
            self._save_cache()

            self.logger.info(f"Added annotation: {annotations} from {t_start} to {t_end}")
            self.logger.info(f"Updated file: {filepath}")

        except Exception as e:
            self.logger.error(f"Error adding annotation: {e}")

    def close(self):
        """
        Explicitly save cache and perform cleanup.
        """
        self._save_cache()
        self.logger.info("DataAnnotator closed and cache saved.")

    def __enter__(self):
        """
        Support context management for safe initialization and cleanup.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensure cache is saved when exiting context.
        """
        self.close()

