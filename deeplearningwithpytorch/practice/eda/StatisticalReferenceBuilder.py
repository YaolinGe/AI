# """
# StatisticalReferenceBuilder module builds a statistical reference for the multi-channel time series segmented data.
#
# Methodology:
#     - 1. Use Gen1CSVHandler to load all the data from the csv files.
#     - 2. Use Segmenter to segment the data.
#     - 3. Return the calculated statistical summary.
#
# Author: Yaolin Ge
# Date: 2024-10-28
# """
# import pandas as pd
# import numpy as np
# from typing import List
# from Gen1CutFileHandler import Gen1CutFileHandler
# from Segmenter.Segmenter import Segmenter
#
#
# class StatisticalReferenceBuilder:
#
#     def __init__(self):
#         self.filenames = None
#         self.gen1_cutfile_handler = Gen1CutFileHandler()
#         self.segmenter = Segmenter()
#
#     def segment_all_data_frames(self, filenames: List[str]) -> None:
#         self.filenames = filenames
#         self.df_segmented_all = []
#         for file in self.filenames:
#             self.gen1_cutfile_handler.process_file(file, resolution_ms=250)
#             df = self.gen1_cutfile_handler.df_sync
#             df_segmented = self.segmenter.segment(df)
#             self.df_segmented_all.append(df_segmented)
#         self.segmented_data_dict = {}
#         for i, df_segmented in enumerate(self.df_segmented_all):
#             for segment_name, segment_df in df_segmented.items():
#                 if segment_name not in self.segmented_data_dict:
#                     self.segmented_data_dict[segment_name] = []
#                 self.segmented_data_dict[segment_name].append(segment_df)
#
#     def calculate_confidence_interval(self):
#         self.segmented_data_dict = {}
#         for segment_name, segment_dfs in self.segmented_data_dict.items():
#             max_len = max(len(df) for df in segment_dfs)
#             max_index = np.argmax([len(df) for df in segment_dfs])
#             padded_df = segment_dfs.copy()
#             for i, df in enumerate(segment_dfs):
#                 if len(df) < max_len:
#                     padded_df[i] = pd.concat([df, pd.DataFrame({col: df.iloc[-1][col] for col in df.columns}, index=range(max_len - len(df)))])
#             padded_df_numpy = np.array([df.to_numpy() for df in padded_df])
#             average = np.mean(padded_df_numpy, axis=0)
#             std = np.std(padded_df_numpy, axis=0)
#             df_average = pd.DataFrame(average, columns=padded_df[0].columns)
#             df_average['timestamp'] = padded_df[max_index]['timestamp']
#             df_std = pd.DataFrame(std, columns=padded_df[0].columns)
#             df_std['timestamp'] = padded_df[max_index]['timestamp']
#             self.segmented_data_dict[segment_name] = {
#                 'average': df_average,
#                 'std': df_std
#             }
#             print(f"Segment {segment_name} has been calculated.")
#         return self.segmented_data_dict
#



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
from Segmenter.Segmenter import Segmenter


class StatisticalReferenceBuilder:
    def __init__(self):
        self.gen1_cutfile_handler = Gen1CutFileHandler()
        self.segmenter = Segmenter()
        self.segmented_data_dict = {}

    def build_statistical_reference(self, filenames: List[str]) -> dict:
        """
        Build a statistical reference for the multi-channel time series segmented data.

        Args:
            filenames (List[str]): List of CSV file names.

        Returns:
            dict: A dictionary containing the average and standard deviation for each segment.
        """
        for filename in filenames:
            self.gen1_cutfile_handler.process_file(filename, resolution_ms=250)
            df = self.gen1_cutfile_handler.df_sync
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
            timestamp_common = segment_data['data'][np.argmax([len(df) for df in segment_data['data']])]['timestamp']
            df_average = pd.DataFrame(np.mean(padded_data_array, axis=0))
            df_average.columns = segment_data['data'][0].columns[1:]
            df_average['timestamp'] = timestamp_common
            df_std = pd.DataFrame(np.std(padded_data_array, axis=0))
            df_std.columns = segment_data['data'][0].columns[1:]
            df_std['timestamp'] = timestamp_common
            self.segmented_data_dict[segment_name]['average'] = df_average
            self.segmented_data_dict[segment_name]['std'] = df_std

        return self.segmented_data_dict
