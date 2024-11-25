"""
InCutDetector module adds extra columns to the dataframe to mark if the timestamp is in cut or out of cut

Created on 2024-11-22
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import numpy as np
import pandas as pd
from scipy.signal import convolve



class InCutDetector:
    def __init__(self):
        """

        """

    def process_incut(self, df: pd.DataFrame = None, window_size: int = 20) -> None:
        """
        Detect the incut in-place
        """
        if df is None:
            return
        df_criterion = df[['timestamp', 'load', 'deflection']].copy()
        df_criterion['incut'] = False
        df_criterion['criterion'] = df_criterion['load'] + abs(df_criterion['deflection'])

        threshold = np.percentile(df_criterion['criterion'], 25)
        df_criterion['incut'] = df_criterion['criterion'] > threshold

        window_size = window_size
        kernel = np.ones(window_size)
        majority_votes = convolve(df_criterion['incut'].astype(int), kernel, mode='same')
        majority_votes = (majority_votes > window_size / 2).astype(bool)
        df_criterion['incut'] = majority_votes

        # self.visualizer.lineplot(df_criterion, line_color="white", line_width=.5, use_plotly=False).show()
        df['incut'] = df_criterion['incut']
