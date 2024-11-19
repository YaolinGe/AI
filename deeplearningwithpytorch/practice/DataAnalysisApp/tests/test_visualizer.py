from unittest import TestCase
import pandas as pd
import numpy as np
import os
from Visualizer import Visualizer
from DataHandler import DataHandler


class TestVisualizer(TestCase):
    def setUp(self) -> None:
        self.visualizer = Visualizer()
        self.dataHandler = DataHandler()
        folder_path = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin"
        files = os.listdir(folder_path)
        filenames = [file[:-4] for file in files if file.endswith('.cut')]
        filepath = os.path.join(folder_path, filenames[0])
        self.dataHandler.load_synchronized_data(filepath)

    def test_plot_data(self):
        fig = self.visualizer.lineplot(self.dataHandler.df_sync, title='Time Series Data')
        # self.assertEqual(fig.get_size_inches().tolist(), [14, 10])
        print("hel")
        fig.show()

