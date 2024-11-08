"""
CutFileMerger will merge all the cut files to one long time series data frame.

Created on 2024-11-7
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
from typing import List
import pandas as pd
from tqdm import tqdm
from CutFileHandler import CutFileHandler


class CutFileMerger:
    def __init__(self, is_gen2: bool=True):
        self._cut_file_handler = CutFileHandler(is_gen2=is_gen2)
        self.df_merged = None

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
        self.df_merged = pd.DataFrame()
        time_start = 0
        for filename in tqdm(filenames):
            self._cut_file_handler.process_file(filename, resolution_ms=resolution_ms)
            df_to_merge = self._cut_file_handler.df_sync
            df_to_merge['timestamp'] = df_to_merge['timestamp'] + time_start
            time_start = df_to_merge['timestamp'].iloc[-1] + resolution_ms / 1000
            self.df_merged = pd.concat([self.df_merged, self._cut_file_handler.df_sync], ignore_index=True)
            self.df_merged.to_csv(filepath, index=False)
            print(f"Merged {filename} to the merged data frame.")
