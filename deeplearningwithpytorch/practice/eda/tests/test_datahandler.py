from unittest import TestCase
import pandas as pd
import numpy as np
import os
from DataHandler import DataHandler
from Gen1CSVHandler import Gen1CSVHandler
from Visualizer import Visualizer


class TestDataHandler(TestCase):

    def setUp(self) -> None:
        self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
        self.gen1CSVHandler = Gen1CSVHandler(self.filePath)
        self.dataHandler = DataHandler(self.gen1CSVHandler.df_sync)
        self.visualizer = Visualizer()

    def test_cropping_data(self) -> None:
        df_cropped = self.dataHandler.crop_data(0.0, 140)
        fig = self.visualizer.lineplot(df_cropped, use_plotly=True, line_width=.5, opacity=.5)
        fig.show()
        print("he")



