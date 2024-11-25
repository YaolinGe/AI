"""
Segmenter module automatically segments the multi-channel time series data using BreakPointDetector.

Author: Yaolin Ge
Date: 2024-10-29
"""
from typing import Dict
import pandas as pd
from Segmenter.BreakPointDetector import BreakPointDetector
from Logger import Logger


class Segmenter:
    _logger = Logger()

    def __init__(self):
        pass

    @staticmethod
    def segment(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segment the multi-channel time series data.
        """
        df_segmented = {}
        signal = df.iloc[:, 1:].to_numpy()
        result = BreakPointDetector.fit(signal, pen=100000, model_type="Pelt", model="l1", min_size=1, jump=1)
        Segmenter._logger.info(f"Detected breakpoints: {result}")
        for i in range(0, len(result), 2):
            segment_name = f"segment_{i // 2}"
            if i + 1 >= len(result):
                continue
            df_segmented[segment_name] = df.iloc[result[i]:result[i + 1], :]
        return df_segmented
