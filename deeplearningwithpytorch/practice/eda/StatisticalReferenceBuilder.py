"""
StatisticalReferenceBuilder module builds a statistical reference for the multi-channel time series segmented data.

Methodology:
    - 1. Use Gen1CSVHandler to load all the data from the csv files.
    - 2. Use Segmenter to segment the data.
    - 3. Return the calculated statistical summary.

Author: Yaolin Ge
Date: 2024-10-28
"""
import pandas as pd
import numpy as np
from typing import List
from Gen1CutFileHandler import Gen1CutFileHandler
from Segmenter.Segmenter import Segmenter


class StatisticalReferenceBuilder:

    def __init__(self):
        self.filenames = None
        self.gen1_cutfile_handler = Gen1CutFileHandler()
        self.segmenter = Segmenter()

    def segment_all_data_frames(self, filenames: List[str]) -> None:
        self.filenames = filenames
        self.df_segmented_all = []
        for file in self.filenames:
            self.gen1_cutfile_handler.process_file(file, resolution_ms=250)
            df = self.gen1_cutfile_handler.df_sync
            df_segmented = self.segmenter.segment(df)
            self.df_segmented_all.append(df_segmented)
        self.segmented_data_dict = {}
        for i, df_segmented in enumerate(self.df_segmented_all):
            for segment_name, segment_df in df_segmented.items():
                if segment_name not in self.segmented_data_dict:
                    self.segmented_data_dict[segment_name] = []
                self.segmented_data_dict[segment_name].append(segment_df)

    def calculate_confidence_interval(self):
        # self.segmented_data_dict = {}
        for segment_name, segment_dfs in self.segmented_data_dict.items():
            max_len = max(len(df) for df in segment_dfs)
            max_index = np.argmax([len(df) for df in segment_dfs])
            padded_df = segment_dfs.copy()
            for i, df in enumerate(segment_dfs):
                if len(df) < max_len:
                    padded_df[i] = pd.concat([df, pd.DataFrame({col: df.iloc[-1][col] for col in df.columns}, index=range(max_len - len(df)))])
            padded_df_numpy = np.array([df.to_numpy() for df in padded_df])

            print("hello")
            padded_df_numpy
            self.segmented_data_dict[segment_name] = padded_df
