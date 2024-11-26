from unittest import TestCase
import pandas as pd
import numpy as np
import os
from Visualizer import Visualizer


class TestVisualizer(TestCase):
    def setUp(self) -> None:
        self.visualizer = Visualizer()
        self.df = pd.read_csv(os.path.join(r"datasets", "df_disk1.csv"))

    def test_plot_data(self):
        self.visualizer.lineplot_with_rect(self.df, t_start=50, t_end=5000, line_color="black", line_width=.5,
                                           use_plotly=True).show()

