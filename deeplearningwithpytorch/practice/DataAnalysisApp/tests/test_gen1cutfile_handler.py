from unittest import TestCase
import pandas as pd
from Gen1CutFileHandler import Gen1CutFileHandler
from Visualizer import Visualizer


class TestGen1CSVHandler(TestCase):

    def setUp(self) -> None:
        self.gen1CutFileHandler = Gen1CutFileHandler()
        self.filePath = r"C:\Data\Gen1CutFile\SilentTools_00410_20211130-143236.cut"
        self.gen1CutFileHandler.process_file(self.filePath, resolution_ms=500)
        self.visualizer = Visualizer()

    def test_sync_data(self):
        fig = self.visualizer.lineplot(self.gen1CutFileHandler.df_sync, line_color="white", use_plotly=False)
        fig.show()
        self.gen1CutFileHandler.print_load_times()

    def test_get_raw_data(self):
        df_accelerometer = self.gen1CutFileHandler.raw_accelerometer
        fig = self.visualizer.lineplot(df_accelerometer, line_color="white", use_plotly=False)
        fig.show()
        df_strain0 = self.gen1CutFileHandler.raw_strain0
        df_strain1 = self.gen1CutFileHandler.raw_strain1
        ddf = pd.concat([df_strain0, df_strain1], axis=1)
        ddf.columns = ['timestamp', 'strain0', 'timestamp', 'strain1']
        ddf = ddf.loc[:, ~ddf.columns.duplicated()]
        fig = self.visualizer.lineplot(ddf, line_color="white", use_plotly=False)
        fig.show()
