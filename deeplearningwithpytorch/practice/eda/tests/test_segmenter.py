from unittest import TestCase
import pandas as pd
import numpy as np
import os
import gc
import ruptures as rpt
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional
from Gen1CSVHandler import Gen1CSVHandler
from Segmenter import Segmenter
from Visualizer import Visualizer


class TestSegmenter(TestCase):

        def setUp(self) -> None:
            self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
            self.gen1CSVHandler = Gen1CSVHandler()
            self.visualizer = Visualizer()

        def test_segment_data(self) -> None:
            self.gen1CSVHandler.process_file(self.filePath, resolution_ms=100)
            df = self.gen1CSVHandler.df_sync
            # fig = self.visualizer.lineplot(df, line_color="white", line_width=.5, use_plotly=False, text_color="white")
            # fig.show()
            signal = df.iloc[:, 1:].to_numpy()
            # self.segmenter = Segmenter(model_type="BottomUp", model="l1")
            # self.segmenter = Segmenter(model_type="Pelt", model="l2")
            self.segmenter = Segmenter(model_type="Binseg", model="l2", min_size=50, jump=5)
            result = self.segmenter.fit(signal, pen=5000000000)
            # fig = self.segmenter.plot_results()
            fig = self.visualizer.segmentplot(df, result, line_color="black", use_plotly=True)
            fig.show()
            # fig.write_html("temp_figure.html", auto_open=True)
            print("he")
