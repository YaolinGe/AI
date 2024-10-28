"""
Segmenter module for the multi-channel time series data segmentation.

Author: Yaolin Ge
Date: 2024-10-28
"""
import ruptures as rpt


class Segmenter:
    def __init__(self, model_type='BottomUp', **model_params):
        """
        Initialize the Segmenter with a specified model and parameters.

        Parameters:
        model_type : str
            Type of segmentation model to use (e.g., 'Binseg', 'Pelt', 'Window', 'Dynp', 'BottomUp').
        model_params : dict
            Additional parameters for the chosen segmentation model.
        """
        self.model_type = model_type
        self.model_params = model_params
        self.bkps = None
        self.signal = None

    def fit(self, signal, pen=500, n_bkps=None):
        """
        Fit the segmentation model to the signal.

        Parameters:
        signal : np.array
            The signal data to segment.
        pen : float
            Penalty value used in the prediction (if n_bkps is None).
        n_bkps : int or None
            Number of breakpoints (if known, overrides pen).

        Returns:
        list of breakpoints.
        """
        self.signal = signal
        model_class = getattr(rpt, self.model_type)
        algo = model_class(**self.model_params).fit(signal)

        # Predict the breakpoints
        if n_bkps:
            self.bkps = algo.predict(n_bkps=n_bkps)
        else:
            self.bkps = algo.predict(pen=pen)

        return self.bkps[:-1]
