"""
Test if incut_detector works as expected.
"""

import pandas as pd
from unittest import TestCase
from InCutDetector import InCutDetector
from Visualizer import Visualizer


class TestInCutDetector(TestCase):

    def setUp(self):
        # self.filepath = r"C:\Data\MissyDataSet\Missy_Disc1\Cutfiles\CoroPlus_240918-114306.cut"
        # self.cut_file_handler = CutFileHandler(is_gen2=True, debug=False)
        self.filepath = r"datasets\df_disk1.csv"
        # self.df = pd.read_csv(self.filepath)[:5000]
        self.df = pd.read_csv(self.filepath)[10000:15000].reset_index()
        self.incut_detector = InCutDetector()
        self.visualizer = Visualizer()

    def test_incut_detection(self):
        self.incut_detector.process_incut(self.df, window_size=20)
        self.visualizer.lineplot(self.df, line_color="white", line_width=.5, use_plotly=False, incut=True).show()
        self.df