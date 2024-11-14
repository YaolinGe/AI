from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from CutFileHandler import CutFileHandler
from Segmenter.BreakPointDetector import BreakPointDetector
from Visualizer import Visualizer


class TestProcessedDataHandler(TestCase):
    def setUp(self):
        self.breakpointDetector = BreakPointDetector()
        self.visualizer = Visualizer()
        self.zone_colors = ['#FFE5E5', '#E5FFE5', '#E5E5FF', '#FFFFE5']  # Light red, green, blue, yellow

    def test_gen2_cutfile_processing(self):
        self.filepath = r"C:\Data\MissyDataSet\Missy_Disc2\CutFiles\CoroPlus_241008-133957.cut"
        self.cutfile_handler = CutFileHandler(is_gen2=True, debug=False)
        self.cutfile_handler.process_file(self.filepath, resolution_ms=500)
        fig = self.visualizer.lineplot(self.cutfile_handler.df_sync, line_color="black", line_width=.5, use_plotly=False)
        fig.show()
        fig.show()

    # def test_gen1_cut_file_processing(self):
    #     self.filepath = r"C:\Users\nq9093\Downloads\JorgensData\Heat Treated HRC46_SS2541_TR-DC1304-F 4415.cut"
    #     self.cutfile_handler = CutFileHandler()
    #     self.cutfile_handler.process_file(self.filepath, resolution_ms=100)
    #     fig = self.visualizer.lineplot(self.cutfile_handler.df_sync, line_color="black", line_width=.5, use_plotly=False)
    #     fig.show()
    #     fig.show()

    # def test_data_segmentation(self) -> None:
    #     signal = self.cutfile_handler.get_synchronized_data()
    #     signal = self.cutfile_handler.df_sync.iloc[:, 1:].to_numpy()
    #     pass
