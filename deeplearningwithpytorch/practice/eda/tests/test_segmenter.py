from unittest import TestCase
from Segmenter import Segmenter
import pandas as pd
import numpy as np
import os
import gc
import ruptures as rpt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional
from Gen1CSVHandler import Gen1CSVHandler
from DataHandler import DataHandler
from Visualizer import Visualizer


class TestSegmenter(TestCase):

        def setUp(self) -> None:
            self.filePath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin\SilentTools_00410_20211130-143236.cut"
            self.gen1CSVHandler = Gen1CSVHandler(self.filePath)
            self.dataHandler = DataHandler(self.gen1CSVHandler.df_sync)
            self.visualizer = Visualizer()
            # self.segmenter = Segmenter(use_gpu=True, n_threads=5)
            self.segmenter = Segmenter(model="l2", n_jobs=4)

        def test_segment_data(self) -> None:
            df = self.dataHandler.crop_data(25, 50)
            signal = df.iloc[:, 1:].to_numpy()
            my_bkps = self.segmenter.fit_predict(signal, pen=500)
            self.plot_signal_with_bkps(signal, my_bkps, save_path="test.png")
            print("he")

        def plot_signal_with_bkps(self, signal: np.ndarray, bkps: List[int], save_path: Optional[str] = None):
            """
            Plot signal with change points and optionally save to file
            """
            plt.clf()  # Clear any existing plots
            n_dims = signal.shape[1]
            fig, axes = plt.subplots(n_dims, 1, figsize=(15, 3 * n_dims))

            if n_dims == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.plot(signal[:, i])
                for k in bkps:
                    ax.axvline(x=k, color='red', alpha=0.5)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close(fig)
            else:
                plt.show()
            plt.close('all')
            gc.collect()