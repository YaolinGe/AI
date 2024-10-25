# """
# Segmenter module handles the segmentation of multi-channel signals
#
# Author: Yaolin Ge
# Date: 2024-10-25
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import ruptures as rpt
#
#
# class Segmenter:
#     def __init__(self):
#         pass
#
#     def run(self, df: pd.DataFrame) -> None:
#         pass
#
#     def fit(self, signal, model, pen=500, n_bkps=None):
#         """
#         Fit the segmentation model to the signal.
#
#         Parameters:
#         pen : float
#             Penalty value used in the prediction (used when n_bkps is None).
#         n_bkps : int or None
#             Number of breakpoints (if known, overrides pen).
#
#         Returns:
#         list of breakpoints.
#         """
#         # Initialize the model
#         # algo = rpt.Binseg(model=model).fit(signal)
#         # algo = rpt.Pelt(model=model).fit(signal)
#         # algo = rpt.Window(width=2, model=model).fit(signal)
#         # algo = rpt.Dynp(model=model).fit(signal)
#         algo = rpt.BottomUp(model=model).fit(signal)
#
#         # Predict the breakpoints
#         if n_bkps:
#             self.bkps = algo.predict(n_bkps=n_bkps)
#         else:
#             self.bkps = algo.predict(pen=pen)
#
#         return self.bkps
#
#     def plot_results(self):
#         """
#         Plot the signal with the detected breakpoints.
#         """
#         n_channels = self.signal.shape[1]
#         fig, axes = plt.subplots(n_channels, 1, figsize=(10, 8))
#
#         if n_channels == 1:
#             axes = [axes]  # Ensure axes is iterable for a single channel
#
#         for i, ax in enumerate(axes):
#             ax.plot(self.signal[:, i], label=f'Channel {i + 1}')
#             for bkpt in self.bkps:
#                 ax.axvline(x=bkpt, color='red', linestyle='--')
#             ax.legend(loc='best')
#
#         plt.tight_layout()
#         plt.show()
#
#
# # Example usage
# if __name__ == "__main__":
#     # Generate synthetic signal for testing (multi-channel)
#     n = 150000  # 150k samples
#     n_channels = 7
#     n_bkps, sigma = 7, 5
#     signal, bkps = rpt.pw_constant(n, n_bkps, noise_std=sigma, n_features=n_channels)
#
#     # Instantiate the segmentation class and fit the model
#     segmenter = Segmenter(signal)
#     detected_bkps = segmenter.fit(pen=500)
#
#     # Plot the results
#     segmenter.plot_results()


"""
Enhanced signal segmentation system with flexible model selection and zone marking.

Author: Yaolin Ge
Date: 2024-10-25
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import ruptures as rpt


class SegmentationModel(Enum):
    """Available segmentation models"""
    BINSEG = "binseg"
    PELT = "pelt"
    WINDOW = "window"
    DYNP = "dynp"
    BOTTOMUP = "bottomup"


@dataclass
class SegmentationZone:
    """Data class representing a segmented zone"""
    start_idx: int
    end_idx: int
    channel_means: np.ndarray
    channel_stds: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'channel_means': self.channel_means.tolist(),
            'channel_stds': self.channel_stds.tolist()
        }


class SegmentationModelFactory:
    """Factory for creating segmentation model instances"""

    @staticmethod
    def create_model(model_type: SegmentationModel, **kwargs) -> rpt.base.BaseEstimator:
        model_map = {
            SegmentationModel.BINSEG: rpt.Binseg,
            SegmentationModel.PELT: rpt.Pelt,
            SegmentationModel.WINDOW: rpt.Window,
            SegmentationModel.DYNP: rpt.Dynp,
            SegmentationModel.BOTTOMUP: rpt.BottomUp
        }

        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = model_map[model_type]
        return model_class(**kwargs)


class SegmentationResult:
    """Class to store and manage segmentation results"""

    def __init__(self, signal: np.ndarray, breakpoints: List[int]):
        self.signal = signal
        self.breakpoints = breakpoints
        self.zones = self._create_zones()

    def _create_zones(self) -> List[SegmentationZone]:
        """Create zones from breakpoints with statistics"""
        zones = []
        start_idx = 0

        for end_idx in self.breakpoints:
            segment = self.signal[start_idx:end_idx]
            zones.append(SegmentationZone(
                start_idx=start_idx,
                end_idx=end_idx,
                channel_means=np.mean(segment, axis=0),
                channel_stds=np.std(segment, axis=0)
            ))
            start_idx = end_idx

        return zones

    def get_zone_boundaries(self) -> List[Tuple[int, int]]:
        """Get list of zone boundaries"""
        return [(zone.start_idx, zone.end_idx) for zone in self.zones]

    def get_zone_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for each zone"""
        return [zone.to_dict() for zone in self.zones]


class Segmenter:
    """Main class for signal segmentation"""

    def __init__(self, model_type: SegmentationModel = SegmentationModel.BOTTOMUP):
        self.model_type = model_type
        self.model_factory = SegmentationModelFactory()
        self.result: Optional[SegmentationResult] = None

    def run(self, df: pd.DataFrame, model: str = "l2", pen: float = 500,
            n_bkps: Optional[int] = None) -> SegmentationResult:
        """
        Run segmentation on the input DataFrame.

        Args:
            df: Input DataFrame
            model: Cost model ("l1", "l2", "rbf", etc.)
            pen: Penalty term for changepoint detection
            n_bkps: Number of breakpoints (optional)

        Returns:
            SegmentationResult object containing zones and statistics
        """
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            signal = df.iloc[:, 1:].values
        else:
            signal = df.values

        # Create and fit the model
        algo = self.model_factory.create_model(
            self.model_type,
            model=model
        ).fit(signal)

        # Detect breakpoints
        bkps = algo.predict(n_bkps=n_bkps) if n_bkps else algo.predict(pen=pen)

        # Create and store result
        self.result = SegmentationResult(signal, bkps)
        return self.result

    def get_zones(self) -> List[SegmentationZone]:
        """Get detected zones"""
        if self.result is None:
            raise ValueError("Must run segmentation first")
        return self.result.zones

    def get_breakpoints(self) -> List[int]:
        """Get detected breakpoints"""
        if self.result is None:
            raise ValueError("Must run segmentation first")
        return self.result.breakpoints


class SegmentationVisualizer:
    """Separate class for visualization of segmentation results"""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize

    def plot_segmentation(self, result: SegmentationResult,
                          channel_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
        """Plot segmentation results"""
        import matplotlib.pyplot as plt

        signal = result.signal
        n_channels = signal.shape[1]

        # Create channel names if not provided
        if channel_names is None:
            channel_names = [f'Channel {i + 1}' for i in range(n_channels)]

        # Create subplots
        fig, axes = plt.subplots(n_channels, 1, figsize=self.figsize)
        if n_channels == 1:
            axes = [axes]

        # Plot each channel
        for i, ax in enumerate(axes):
            # Plot signal
            ax.plot(signal[:, i], label=channel_names[i], alpha=0.7)

            # Plot breakpoints
            for bkpt in result.breakpoints:
                ax.axvline(x=bkpt, color='red', linestyle='--', alpha=0.5)

            # Add zone statistics
            for zone in result.zones:
                mean = zone.channel_means[i]
                std = zone.channel_stds[i]
                ax.axhline(y=mean,
                           xmin=zone.start_idx / len(signal),
                           xmax=zone.end_idx / len(signal),
                           color='green', alpha=0.5)

            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic signal
    n = 150000
    n_channels = 7
    n_bkps, sigma = 7, 5
    signal, true_bkps = rpt.pw_constant(n, n_bkps, noise_std=sigma)
    df = pd.DataFrame(signal, columns=[f'Channel_{i + 1}' for i in range(n_channels)])

    # Create segmenter with desired model
    segmenter = Segmenter(model_type=SegmentationModel.BOTTOMUP)

    # Run segmentation
    result = segmenter.run(df, pen=500)

    # Access results
    print("Breakpoints:", segmenter.get_breakpoints())
    print("\nZone Statistics:")
    for i, zone in enumerate(segmenter.get_zones(), 1):
        print(f"\nZone {i}:")
        print(f"Start: {zone.start_idx}, End: {zone.end_idx}")
        print(f"Channel Means: {zone.channel_means}")

    # Visualize results
    visualizer = SegmentationVisualizer(figsize=(15, 12))
    visualizer.plot_segmentation(result)