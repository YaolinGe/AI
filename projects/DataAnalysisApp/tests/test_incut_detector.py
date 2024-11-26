"""
Test if incut_detector works as expected.
"""

import pandas as pd
import os
from unittest import TestCase
from InCutDetector import InCutDetector
from Visualizer import Visualizer


class TestInCutDetector(TestCase):

    def setUp(self):
        # self.filepath = r"C:\Data\MissyDataSet\Missy_Disc1\Cutfiles\CoroPlus_240918-114306.cut"
        # self.cut_file_handler = CutFileHandler(is_gen2=True, debug=False)
        self.filepath = r"datasets\df_disk1.csv"
        # self.df = pd.read_csv(self.filepath)[:5000]
        # self.df = pd.read_csv(self.filepath)[10000:15000].reset_index()
        self.df = pd.read_csv(self.filepath)
        self.incut_detector = InCutDetector()
        self.visualizer = Visualizer()

    def test_incut_detection(self):
        self.incut_detector.process_incut(self.df, window_size=20)
        timestamps = pd.read_csv(os.path.join(".incut", "incut.csv"))

        df_annotations = pd.read_csv(os.path.join("annotations", "df_disk1_annotation.csv"))

        timestamps.columns = ['TStart', 'TEnd']
        # timestamps['Annotations'] = 'InCut'

        def merge_intervals(df_annotations, timestamps):
            # Step 1: Prepare data by concatenating the two DataFrames
            timestamps['Annotations'] = 'InCut'  # Ensure all timestamps are labeled 'InCut'
            df_incut = df_annotations[df_annotations['Annotations'] == 'InCut']
            combined_df = pd.concat([df_incut, timestamps], ignore_index=True)

            # Step 2: Sort by 'TStart' to make merging easier
            combined_df = combined_df.sort_values(by='TStart').reset_index(drop=True)

            # Step 3: Initialize the list to hold merged intervals
            merged_intervals = []

            # Step 4: Iterate through each row in the sorted combined DataFrame
            for _, row in combined_df.iterrows():
                if not merged_intervals:
                    merged_intervals.append(row)
                    continue

                # Compare with the last added interval
                last_interval = merged_intervals[-1]

                # Check for overlap
                if row['TStart'] <= last_interval['TEnd']:
                    # Overlapping intervals; merge them
                    last_interval['TEnd'] = max(last_interval['TEnd'], row['TEnd'])

                    # # Priority of annotations (adjust according to your preference)
                    # priority = {"Anomaly": 3, "InCut": 2, "Normal": 1}
                    # if priority.get(row['Annotations'], 0) > priority.get(last_interval['Annotations'], 0):
                    #     last_interval['Annotations'] = row['Annotations']
                else:
                    # Non-overlapping interval; add as new
                    merged_intervals.append(row)

            # Step 5: Convert the merged intervals back to a DataFrame
            merged_df = pd.DataFrame(merged_intervals)

            # Step 6: Drop exact duplicate rows if necessary
            merged_df = merged_df.drop_duplicates(subset=['TStart', 'TEnd', 'Annotations']).reset_index(drop=True)
            merged_df = pd.concat([merged_df, df_annotations[df_annotations['Annotations'] != 'InCut']], ignore_index=True)
            merged_df = merged_df.sort_values(by=['TStart', 'TEnd'])

            return merged_df

        merged_df = merge_intervals(df_annotations, timestamps)

        # Visualize the merged intervals
        import matplotlib.pyplot as plt
        plt.figure(figsize=(100, 10), dpi=300)
        plt.plot(self.df['timestamp'], self.df['load'])
        for _, row in merged_df.iterrows():
            if row['Annotations'] == 'InCut':
                plt.gca().axvspan(row['TStart'], row['TEnd'], alpha=0.5, color='red')
            else:
                plt.gca().text(row['TStart'], self.df['load'].max() * .5, row['Annotations'], color='black', fontsize=16, rotation=90, verticalalignment='bottom')

        plt.xlabel("Timestamp")
        plt.ylabel("Load")
        plt.title("InCut Detection")
        plt.xlim([0, self.df['timestamp'].max()])
        # plt.show()
        plt.savefig("incut_detection.png")
        plt.close("all")
        merged_df.to_csv(os.path.join("annotations", "df_disk1_annotation_merged.csv"), index=False)
        self.visualizer.lineplot_with_rect(self.df, t_start=50, t_end=5000, line_color="black", line_width=.5,)

