"""
InCutDetector module adds extra columns to the dataframe to mark if the timestamp is in cut or out of cut

Created on 2024-11-22
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import numpy as np
import pandas as pd
import os
from scipy.signal import convolve
from Logger import Logger


class InCutDetector:
    def __init__(self):
        """

        """
        self.logger = Logger()

    def process_incut(self, df: pd.DataFrame = None, window_size: int = 20) -> None:
        """
        Detect the incut in-place
        """
        if df is None:
            return

        load = df['load'].values
        deflection = df['deflection'].values
        criterion = load + abs(deflection)

        threshold = np.percentile(criterion, 25)
        incut = criterion > threshold

        window_size = window_size
        kernel = np.ones(window_size)
        majority_votes = convolve(incut.astype(int), kernel, mode='same')
        majority_votes = (majority_votes > window_size / 2).astype(bool)
        df['incut'] = majority_votes

        self.logger.info(f"Number of incut points: {np.sum(majority_votes)}")

        incut_changes = df['incut'].ne(df['incut'].shift()).cumsum()
        indices = df['incut'].groupby(incut_changes).apply(
            lambda x: (x.index[0], x.index[-1]) if x.iloc[0] else None
        ).dropna()
        timestamps = []
        for start, end in indices:
            timestamps.append((df['timestamp'].iloc[start], df['timestamp'].iloc[end]))

        timestamps = pd.DataFrame(timestamps, columns=['start', 'end'])
        timestamps.to_csv(os.path.join(".incut", "incut.csv"), index=False)
        self.logger.info(f"Saved incut timestamps to .incut/incut.csv")

