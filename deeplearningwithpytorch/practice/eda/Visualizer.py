import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches


class Visualizer:

    def __init__(self):
        pass

    def plotly_data(self, df: pd.DataFrame, line_color: str='white', t_start: float=None, t_end: float=None) -> go.Figure:
        """
        Plot the time series data using Plotly with optional time range highlighting.

        Args:
            df: DataFrame containing the time series data
            line_color: Color of the plotted lines (default: 'white')
            t_start: Start time for highlighting (default: None)
            t_end: End time for highlighting (default: None)

        Returns:
            go.Figure: The created Plotly figure object
        """
        fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        columns = ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']

        for i, column in enumerate(columns):
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[column], mode='lines', name=column,
                                    line=dict(color=line_color, width=1), showlegend=False), row=i+1, col=1)
            fig.update_yaxes(title_text=column, row=i+1, col=1)

            # Add highlighting if t_start and t_end are provided
            if t_start is not None and t_end is not None:
                fig.add_vrect(
                    x0=t_start, x1=t_end,
                    fillcolor="yellow", opacity=0.3, line_width=0,
                    row=i+1, col=1
                )

        fig.update_layout(
            height=600,
            width=1000,
            xaxis7=dict(title="timestamp"),
        )

        return fig

    def plot_data(self, df: pd.DataFrame, line_color: str = 'black',
              figsize: tuple = (20, 8), t_start: float = None, t_end: float = None) -> plt.Figure:
        """
        Plot the time series data using matplotlib with GridSpec and highlight a specific time range.

        Args:
            df: DataFrame containing the time series data
            line_color: Color of the plotted lines (default: 'black')
            figsize: Size of the figure as (width, height) in inches (default: (20, 8))
            t_start: Start time for highlighting (default: None)
            t_end: End time for highlighting (default: None)

        Returns:
            plt.Figure: The created figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(7, 1, figure=fig, hspace=0.3)

        columns = ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']

        t_min, t_max = df['timestamp'].min(), df['timestamp'].max()

        for i, column in enumerate(columns):
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(df['timestamp'], df[column], linewidth=1, color=line_color)
            ax.set_ylabel(column)
            if i < len(columns) - 1:
                ax.tick_params(axis='x', labelsize=0)
            else:
                ax.set_xlabel('timestamp')
            ax.set_xlim(t_min, t_max)

            # Add highlighting rectangle if t_start and t_end are provided
            if t_start is not None and t_end is not None:
                highlight_color = 'yellow'
                alpha = 0.3
                rect = patches.Rectangle((t_start, ax.get_ylim()[0]), t_end - t_start, 
                                        ax.get_ylim()[1] - ax.get_ylim()[0],
                                        linewidth=0, facecolor=highlight_color, alpha=alpha)
                ax.add_patch(rect)
        return fig

    def plot_train_val_losses(self, train_losses, val_losses) -> plt.Figure:
        """
        Plot the training and validation losses during model training.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        fig, ax = plt.subplots()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Losses')
        ax.legend()
        return fig


















