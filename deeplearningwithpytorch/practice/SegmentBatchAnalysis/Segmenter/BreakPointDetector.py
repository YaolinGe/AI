"""
Segmenter module for the multi-channel time series data segmentation.

Author: Yaolin Ge
Date: 2024-10-28
"""
import ruptures as rpt
import numpy as np
from typing import List, Union, Tuple


class BreakPointDetector:
    def __init__(self):
        pass

    @staticmethod
    def fit(signal: np.ndarray, pen: float = 500, model_type: str = 'BottomUp', **model_params):
        """
        Fit the segmentation model to the signal.

        Parameters:
        signal : np.array
            The signal data to segment.
        pen : float
            Penalty value used in the prediction (if n_bkps is None).
        model_type : str
            Type of segmentation model to use (e.g., 'Binseg', 'Pelt', 'Window', 'Dynp', 'BottomUp').
        model_params : dict
            Additional parameters for the chosen segmentation model.

        Returns:
        list of breakpoints.
        """
        model_class = getattr(rpt, model_type)
        algo = model_class(**model_params).fit(signal)
        bkps = algo.predict(pen=pen)

        return bkps[:-1] # Exclude the last breakpoint

    # @staticmethod
    # def detect_transitions(signal: np.ndarray, breakpoints: List[int],
    #                        window_size: int = 5) -> List[int]:
    #     """
    #     Detect actual transition points by analyzing multi-channel signals around detected breakpoints.
    #
    #     Parameters:
    #     -----------
    #     signal : np.ndarray
    #         The input signal data (shape: [n_samples, n_channels])
    #     breakpoints : List[int]
    #         Initially detected breakpoints
    #     window_size : int
    #         Size of the window to analyze around each breakpoint
    #
    #     Returns:
    #     --------
    #     List[int]
    #         Adjusted breakpoints at the actual transition points
    #     """
    #     adjusted_breakpoints = []
    #
    #     # Ensure signal is 2D
    #     if signal.ndim == 1:
    #         signal = signal.reshape(-1, 1)
    #
    #     for bkp in breakpoints:
    #         # Define window around breakpoint
    #         start_idx = max(0, bkp - window_size)
    #         end_idx = min(len(signal), bkp + window_size)
    #         window = signal[start_idx:end_idx]
    #
    #         if len(window) < 2:
    #             continue
    #
    #         # Calculate differences for each channel
    #         diffs = np.sum(np.abs(np.diff(window, axis=0)), axis=1)
    #
    #         # Find the point of maximum combined change within the window
    #         max_change_idx = np.argmax(diffs)
    #         actual_transition = start_idx + max_change_idx
    #         # reference = np.sum(np.amax(signal, axis=0) - np.amin(signal, axis=0))
    #         # Only include if the total change is significant
    #         # if diffs[max_change_idx] > threshold * reference:
    #         adjusted_breakpoints.append(actual_transition)
    #
    #     return sorted(adjusted_breakpoints)

    # @staticmethod
    # def fit(signal: np.ndarray, pen: float = 500, model_type: str = 'BottomUp',
    #         detect_transitions: bool = True, window_size: int = 5, **model_params) -> Union[List[int], Tuple[List[int], List[int]]]:
    #     """
    #     Fit the segmentation model to the signal and optionally detect transition points.
    #
    #     Parameters:
    #     -----------
    #     signal : np.ndarray
    #         The signal data to segment
    #     pen : float
    #         Penalty value used in the prediction
    #     model_type : str
    #         Type of segmentation model ('Binseg', 'Pelt', 'Window', 'Dynp', 'BottomUp')
    #     detect_transitions : bool
    #         Whether to detect actual transition points
    #     window_size : int
    #         Size of the window for transition detection
    #     threshold : float
    #         Threshold for determining significant changes
    #     model_params : dict
    #         Additional parameters for the chosen segmentation model
    #
    #     Returns:
    #     --------
    #     If detect_transitions=False:
    #         List[int]: breakpoints
    #     If detect_transitions=True:
    #         Tuple[List[int], List[int]]: (original_breakpoints, transition_points)
    #     """
    #     # Validate input
    #     if not isinstance(signal, np.ndarray):
    #         signal = np.array(signal)
    #
    #     # Fit the model
    #     model_class = getattr(rpt, model_type)
    #     algo = model_class(**model_params).fit(signal)
    #     breakpoints = algo.predict(pen=pen)[:-1]  # Exclude the last breakpoint
    #
    #     if not detect_transitions:
    #         return breakpoints
    #
    #     # Detect actual transition points
    #     transition_points = BreakPointDetector.detect_transitions(signal, breakpoints, window_size)
    #
    #     return breakpoints, transition_points
