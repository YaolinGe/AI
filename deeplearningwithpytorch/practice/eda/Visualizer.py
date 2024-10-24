import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Union, Optional, List


class Visualizer:
    def __init__(self):
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def lineplot(self,
                 df: pd.DataFrame,
                 line_color: Union[str, List[str]] = None,
                 line_width: float = 1.0,
                 height_per_plot: float = 150,  # Height in pixels for each subplot
                 plot_width: int = 1200,  # Total width in pixels
                 use_plotly: bool = False,
                 opacity: float = 1.0,
                 exclude_cols: List[str] = None,
                 title: str = "") -> Union[plt.Figure, go.Figure]:
        """
        Create vertically stacked line plots for time series alignment checking.

        Args:
            df: Input DataFrame
            line_color: Color(s) for the lines. Can be single color or list of colors
            line_width: Width of the lines
            height_per_plot: Height of each subplot in pixels
            plot_width: Total width of the plot in pixels
            use_plotly: Whether to use Plotly instead of Matplotlib
            opacity: Opacity of the lines
            exclude_cols: List of column names to exclude from plotting
            title: Title for the entire figure
        """
        # Filter out excluded columns
        if exclude_cols is None:
            exclude_cols = []

        # Identify timestamp column (first column)
        timestamp_col = df.columns[0]
        plot_cols = [col for col in df.columns if col != timestamp_col and col not in exclude_cols]

        if not plot_cols:
            raise ValueError("No columns to plot after excluding timestamp and specified columns.")

        # Number of plots
        n_plots = len(plot_cols)

        # Handle color specification
        if line_color is None:
            colors = self.default_colors
        elif isinstance(line_color, str):
            colors = [line_color] * n_plots
        else:
            colors = line_color

        if use_plotly:
            fig = make_subplots(
                rows=n_plots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.01,  # Minimal spacing between subplots
                subplot_titles=plot_cols
            )

            # Get global y-range for each column for better comparison
            y_ranges = {}
            for col in plot_cols:
                y_ranges[col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'range': df[col].max() - df[col].min()
                }

            for idx, column in enumerate(plot_cols):
                fig.add_trace(
                    go.Scatter(
                        x=df[timestamp_col],
                        y=df[column],
                        mode='lines',
                        name=column,
                        line=dict(color=colors[idx % len(colors)], width=line_width),
                        opacity=opacity
                    ),
                    row=idx + 1,
                    col=1
                )

                # Add y-axis label with min/max values
                y_min = y_ranges[column]['min']
                y_max = y_ranges[column]['max']
                fig.update_yaxes(
                    title_text=f"{column}<br>min: {y_min:.2f}<br>max: {y_max:.2f}",
                    row=idx + 1,
                    col=1,
                    gridcolor='lightgray',
                    griddash='dash'
                )

                # Only show x-axis label for bottom plot
                if idx == n_plots - 1:
                    fig.update_xaxes(title_text=timestamp_col, row=idx + 1, col=1)

                # Add gridlines
                fig.update_xaxes(showgrid=True, gridcolor='lightgray', griddash='dash', row=idx + 1, col=1)

            fig.update_layout(
                height=height_per_plot * n_plots,
                width=plot_width,
                showlegend=False,
                title_text=title,
                title_x=0.5,
                plot_bgcolor='white'
            )

        else:
            # Calculate figure size based on number of plots
            fig_height = n_plots * 2  # Each subplot gets 2 inches
            fig = plt.figure(figsize=(15, fig_height))
            gs = GridSpec(n_plots, 1, figure=fig, hspace=0.3)

            # Get global min/max for x-axis
            t_min, t_max = df[timestamp_col].min(), df[timestamp_col].max()

            for idx, column in enumerate(plot_cols):
                ax = fig.add_subplot(gs[idx, 0])
                ax.plot(
                    df[timestamp_col],
                    df[column],
                    linewidth=line_width,
                    color=colors[idx % len(colors)],
                    alpha=opacity
                )

                # Add y-axis label with min/max values
                y_min, y_max = df[column].min(), df[column].max()
                ax.set_ylabel(f"{column}\nmin: {y_min:.2f}\nmax: {y_max:.2f}")

                ax.set_xlim(t_min, t_max)
                ax.grid(True, linestyle='--', alpha=0.7)

                # Only show x-axis label for bottom plot
                if idx < n_plots - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(timestamp_col)
                    plt.xticks(rotation=45)

            if title:
                fig.suptitle(title, fontsize=12)

            plt.tight_layout()

        return fig