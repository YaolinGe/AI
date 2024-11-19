"""
StatisticalReferenceBuilder module builds a statistical reference for the multi-channel time series segmented data.

Methodology:
    1. Load all data from CSV files using Gen1CSVHandler.
    2. Segment the data using Segmenter.
    3. Pad the segmented data to the maximum length.
    4. Calculate the average and standard deviation for each segment.

Author: Yaolin Ge
Date: 2024-10-28
"""
import pandas as pd
import numpy as np
from typing import List
from Gen1CutFileHandler import Gen1CutFileHandler
from CutFileHandler import CutFileHandler
from Segmenter.Segmenter import Segmenter


class BatchAnalyzer:
    def __init__(self):
        self.gen1_cutfile_handler = Gen1CutFileHandler()
        self.cutfile_handler = CutFileHandler()
        self.segmenter = Segmenter()
        self.segmented_data_dict = {}

    def analyze_batch_cutfiles(self, filenames: List[str], resolution_ms: int, is_gen1: bool = True) -> dict:
        """
        Build a statistical reference for the multi-channel time series segmented data.

        Args:
            filenames (List[str]): A list of filenames to process.
            resolution_ms (int): Resolution in milliseconds.
            is_gen1 (bool): Whether the cut files are in Gen1 format.

        Returns:
            dict: A dictionary containing the average and standard deviation for each segment.
        """
        self.segmented_data_dict = {}
        for filename in filenames:
            if is_gen1:
                self.gen1_cutfile_handler.process_file(filename, resolution_ms=resolution_ms)
                df = self.gen1_cutfile_handler.df_sync
            else:
                self.cutfile_handler.process_file(filename, resolution_ms=resolution_ms)
                # df = self.cutfile_handler.df_sync
                df = self.cutfile_handler.df_sync[
                    ['timestamp', 'x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']]
            df_segmented = self.segmenter.segment(df)

            for segment_name, segment_df in df_segmented.items():
                if segment_name not in self.segmented_data_dict:
                    self.segmented_data_dict[segment_name] = {'data': []}

                self.segmented_data_dict[segment_name]['data'].append(segment_df)

        for segment_name, segment_data in self.segmented_data_dict.items():
            max_length = max(len(df) for df in segment_data['data'])
            padded_data = []

            for df in segment_data['data']:
                if len(df) < max_length:
                    temp_df = pd.DataFrame({col: df.iloc[-1][col] for col in df.columns if col != 'timestamp'},
                                           index=range(max_length - len(df)))
                    padded_df = pd.concat([df.iloc[:, 1:], temp_df])
                else:
                    padded_df = df.iloc[:, 1:]
                padded_data.append(padded_df.to_numpy())

            padded_data_array = np.array(padded_data)
            # Retrieve timestamp values from the longest segment and trim to match max_length if needed
            timestamp_common = segment_data['data'][np.argmax([len(df) for df in segment_data['data']])][
                'timestamp'].values
            timestamp_common = timestamp_common[:max_length]

            # Create df_average and df_std, then insert timestamp as the first column
            df_average = pd.DataFrame(np.mean(padded_data_array, axis=0))
            df_average.columns = segment_data['data'][0].columns[1:]
            df_average['timestamp'] = timestamp_common
            df_average.insert(0, 'timestamp', df_average.pop('timestamp'))

            df_std = pd.DataFrame(np.std(padded_data_array, axis=0))
            df_std.columns = segment_data['data'][0].columns[1:]
            df_std['timestamp'] = timestamp_common
            df_std.insert(0, 'timestamp', df_std.pop('timestamp'))

            self.segmented_data_dict[segment_name]['average'] = df_average
            self.segmented_data_dict[segment_name]['std'] = df_std

        return self.segmented_data_dict
