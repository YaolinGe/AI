from unittest import TestCase
from Segmenter import Segmenter
import pandas as pd
import numpy as np
import os
from Gen1CSVHandler import Gen1CSVHandler
from DataHandler import DataHandler
from Visualizer import Visualizer


class TestSegmenter(TestCase):

        def setUp(self) -> None:
            self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
            self.gen1CSVHandler = Gen1CSVHandler(self.filePath)
            self.dataHandler = DataHandler(self.gen1CSVHandler.df_sync)
            self.visualizer = Visualizer()
            self.segmenter = Segmenter(use_gpu=True, n_threads=5)


        def test_segment_data(self) -> None:
            df = self.dataHandler.crop_data(10, 50)
            # fig = self.visualizer.lineplot(df, use_plotly=False, line_width=.5, opacity=.5)
            # fig.show()
            results = self.segmenter.analyze_all_channels(df, penalty_value=1)
            fig = self.segmenter.plot_results(df, results)
            for channel, data in results.items():
                print(f"\nChangepoints for {channel}:")
                print(f"Number of changepoints: {len(data['changepoints'])}")
                print("Timestamps:", [df['Time'].iloc[cp] for cp in data['changepoints']])
            fig.show()
            print("he")
