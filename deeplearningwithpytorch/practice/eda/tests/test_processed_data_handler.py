from unittest import TestCase
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from ProcessedDataHandler import ProcessedDataHandler
from Segmenter import Segmenter


class TestProcessedDataHandler(TestCase):
    def setUp(self):
        self.filePath = r"C:\Users\nq9093\AppData\Local\Temp\CutFileParser"
        self.processedDataHandler = ProcessedDataHandler(self.filePath)
        self.segmenter = Segmenter()
        self.zone_colors = ['#FFE5E5', '#E5FFE5', '#E5E5FF', '#FFFFE5']  # Light red, green, blue, yellow

    def test_data_segmentation(self):
        signal = self.processedDataHandler.df_sync.iloc[:, 1:].to_numpy()
        # signal = signal[:, :3]
        my_bkps = self.segmenter.fit(signal, model="l2", pen=.4)
        fig = self.plot_signal_with_bkps(signal, my_bkps)
        plt.show()
        plt.show()

    # def plot_signal_with_bkps(self, signal: np.ndarray, bkps: List[int]) -> plt.Figure:
    #     n_dims = signal.shape[1]
    #     fig, axes = plt.subplots(n_dims, 1, figsize=(15, 3 * n_dims))
    #     if n_dims == 1:
    #         axes = [axes]

    #     for i, ax in enumerate(axes):
    #         ax.plot(signal[:, i])
    #         for j in range(len(bkps) - 1):
    #             ax.axvspan(bkps[j], bkps[j + 1], color='red', alpha=0.3, ymin=0, ymax=1)

    #     plt.tight_layout()
    #     return fig
    
    def plot_signal_with_bkps(
        self,
        signal: np.ndarray,
        bkps: List[int],
        figsize: Tuple[int, int] = None,
        title: Optional[str] = None,
        ylabels: Optional[List[str]] = None,
        highlight_zones: bool = True,
        alpha: float = 0.8,
        line_color: str = 'blue',
        show_boundaries: bool = True
    ) -> plt.Figure:
        """
        Plot signal with highlighted zones between breakpoints.
        
        Args:
            signal: Input signal array of shape (n_samples, n_dimensions)
            bkps: List of breakpoint indices
            figsize: Optional figure size tuple (width, height)
            title: Optional title for the entire figure
            ylabels: Optional list of labels for each dimension
            highlight_zones: Whether to highlight zones between breakpoints
            alpha: Transparency of the highlighted zones
            line_color: Color of the signal line
            show_boundaries: Whether to show vertical lines at breakpoints
        
        Returns:
            matplotlib Figure object
        """
        n_dims = signal.shape[1]
        
        # Calculate appropriate figure size if not provided
        if figsize is None:
            figsize = (15, 3 * n_dims)
        
        # Create figure and axes
        fig, axes = plt.subplots(n_dims, 1, figsize=figsize)
        if n_dims == 1:
            axes = [axes]
            
        # Ensure bkps includes start and end points
        full_bkps = [0] + sorted(bkps) + [signal.shape[0]]
        
        # Plot each dimension
        for i, ax in enumerate(axes):
            # Plot the signal line
            ax.plot(signal[:, i], color=line_color, linewidth=1)
            
            if highlight_zones:
                # Highlight zones between breakpoints
                for j in range(1, len(full_bkps) - 1, 2):
                    color = self.zone_colors[j % len(self.zone_colors)]
                    ax.axvspan(
                        full_bkps[j],
                        full_bkps[j + 1],
                        color=color,
                        alpha=alpha,
                        # label=f'Zone {j+1}' if i == 0 else ""
                    )
            if show_boundaries:
                for bkp in bkps:
                    ax.axvline(x=bkp, color='black', linestyle='--', alpha=0.9)
            ax.set_xlim(0, signal.shape[0])
        plt.tight_layout()
        return fig

    def plot_signal_with_probs(
        self,
        signal: np.ndarray,
        bkps: List[int],
        probs: np.ndarray,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot signal with highlighted zones and probability scores.
        
        Args:
            signal: Input signal array
            bkps: List of breakpoint indices
            probs: Array of probability scores for each time point
            figsize: Optional figure size tuple
        """
        n_dims = signal.shape[1]
        if figsize is None:
            figsize = (15, 3 * (n_dims + 1))  # Extra space for probability plot
            
        fig, axes = plt.subplots(n_dims + 1, 1, figsize=figsize)
        
        # Plot signal with zones
        for i in range(n_dims):
            axes[i].plot(signal[:, i], color='blue', linewidth=1)
            
            # Highlight zones between breakpoints
            full_bkps = [0] + sorted(bkps) + [signal.shape[0]]
            for j in range(len(full_bkps) - 1):
                color = self.zone_colors[j % len(self.zone_colors)]
                axes[i].axvspan(
                    full_bkps[j],
                    full_bkps[j + 1],
                    color=color,
                    alpha=0.2
                )
            
            axes[i].set_ylabel(f'Dimension {i+1}')
            axes[i].grid(True, alpha=0.3)
        
        # Plot probability scores
        axes[-1].plot(probs, color='red', label='Change Point Probability')
        axes[-1].fill_between(range(len(probs)), probs, alpha=0.3, color='red')
        axes[-1].set_ylabel('Probability')
        axes[-1].set_xlabel('Time')
        axes[-1].grid(True, alpha=0.3)
        axes[-1].legend()
        
        plt.tight_layout()
        return fig