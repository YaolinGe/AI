from unittest import TestCase
import numpy as np
import os
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
        # self.filepath = r"C:\Data\MissyDataSet\Missy_Disc2\CutFiles\CoroPlus_241008-133957.cut"
        folderpath = r"C:\Data\MissyDataSet\Missy_Disc2\CutFiles"
        figpath = os.path.join(os.getcwd(), "fig")
        files = os.listdir(folderpath)
        files = [os.path.join(folderpath, file) for file in files if file.endswith('.cut')]
        files = files[:3]
        self.cutfile_handler = CutFileHandler(is_gen2=True, debug=False)
        for filepath in files:
            self.cutfile_handler.process_file(filepath, resolution_ms=500)
            # fig = self.visualizer.lineplot(self.cutfile_handler.df_sync, line_color="black", line_width=.5, use_plotly=False)
            fig = self.visualizer.lineplot_with_poi(self.cutfile_handler.df_sync,
                                                    self.cutfile_handler.df_point_of_interests,
                                                    line_color="black", line_width=.5, use_plotly=True,
                                                    text_color="black")
            # fig.show()
            fig.write_html(os.path.join(figpath, os.path.basename(filepath) + ".html"))
            # fig.to_html(os.path.join(figpath, os.path.basename(filepath) + ".html"))
            # fig.close()

