from unittest import TestCase

import pandas as pd

from Gen1CSVHandler import Gen1CSVHandler
from Visualizer import Visualizer


class TestGen1CSVHandler(TestCase):

    def setUp(self) -> None:
        self.gen1CSVHandler = Gen1CSVHandler()
        self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
        self.gen1CSVHandler.process_file(self.filePath)
        self.visualizer = Visualizer()

    def test_sync_data(self):
        fig = self.visualizer.lineplot(self.gen1CSVHandler.df_sync, line_color="black", use_plotly=False)
        fig.show()
        self.gen1CSVHandler.print_load_times()

    def test_get_raw_data(self):
        df_accelerometer = self.gen1CSVHandler.raw_accelerometer
        fig = self.visualizer.lineplot(df_accelerometer, line_color="black", use_plotly=False)
        fig.show()
        df_strain0 = self.gen1CSVHandler.raw_strain0
        df_strain1 = self.gen1CSVHandler.raw_strain1
        ddf = pd.concat([df_strain0, df_strain1], axis=1)
        ddf.columns = ['timestamp', 'strain0', 'timestamp', 'strain1']
        ddf = ddf.loc[:, ~ddf.columns.duplicated()]
        fig = self.visualizer.lineplot(ddf, line_color="black", use_plotly=False)
        fig.show()
