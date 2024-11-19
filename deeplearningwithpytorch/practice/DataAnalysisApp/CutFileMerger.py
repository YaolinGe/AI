"""
CutFileMerger will merge all the cut files to one long time series data frame.

Created on 2024-11-7
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
from typing import List
import pandas as pd
from time import time
import os
from CutFileHandler import CutFileHandler
from Logger import Logger


class CutFileMerger:
    def __init__(self, is_gen2: bool=True):
        self.logger = Logger()
        self._cut_file_handler = CutFileHandler(is_gen2=is_gen2)
        self.df_merged = None
        self.df_point_of_interests_merged = None

    def merge_cut_files(self, filenames: List[str], resolution_ms: int = 250, filepath: str = None) -> None:
        """
        Merge all the cut files to one long time series data frame.

        Args:
            filenames: List of filenames to merge.

        Returns:
            pd.DataFrame: Merged time series data frame.
        """
        filenames.sort()
        if filepath is None:
            filepath = "merged.csv"
            self.logger.warning("No file path provided. Saving to default file path: merged.csv")

        self.df_merged = pd.DataFrame()
        self.df_point_of_interests_merged = pd.DataFrame()
        time_start = 0
        for filename in filenames:
            t1 = time()
            self.logger.info(f"Merging {filename}")
            self._cut_file_handler.process_file(filename, resolution_ms=resolution_ms)
            df_to_merge = self._cut_file_handler.df_sync
            df_to_merge['timestamp'] = df_to_merge['timestamp'] + time_start
            if len(self._cut_file_handler.df_point_of_interests) > 0:
                df_point_of_interests_to_merge = self._cut_file_handler.df_point_of_interests
                df_point_of_interests_to_merge['InCutTime'] = df_point_of_interests_to_merge['InCutTime'] + time_start
                df_point_of_interests_to_merge['OutOfCutTime'] = df_point_of_interests_to_merge['OutOfCutTime'] + time_start
                self.df_point_of_interests_merged = pd.concat([self.df_point_of_interests_merged, df_point_of_interests_to_merge], ignore_index=True)

            self.df_merged = pd.concat([self.df_merged, self._cut_file_handler.df_sync], ignore_index=True)
            time_start = df_to_merge['timestamp'].iloc[-1] + resolution_ms / 1000
            print(f"Merged {filename} to the merged data frame.")
            self.logger.info(f"Merged {filename} to the merged data frame.")
            t2 = time()
            self.logger.info(f"Time taken to merge {filename}: {t2 - t1:.2f} seconds, estimated time to complete: {(t2 - t1) * (len(filenames) - filenames.index(filename)) / 60:.2f} minutes")
        try:
            self.df_merged.to_csv(filepath, index=False)
            self.df_point_of_interests_merged.to_csv(filepath[:-4] + "_POI.csv", index=False)
            self.logger.info(f"Saved merged data frame to {filepath}")
            self.logger.info(f"Saved merged point of interests data frame to {os.path.join(filepath[:-4], '_POI.csv')}")
        except FileNotFoundError:
            self.logger.error(f"File path {filepath} not found.")
