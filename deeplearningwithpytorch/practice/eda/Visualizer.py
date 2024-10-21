import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from typing import Union


class Visualizer:

    def __init__(self):
        self.columns = ['x2g', 'y2g', 'z2g', 'x50g', 'y50g', 'strain0', 'strain1']
        pass

    def lineplot(self, df: pd.DataFrame, line_color: str = 'black', line_width: float = 1.0, figsize: tuple = (20, 8),
                 use_plotly: bool = False, opacity: float = 1.0) -> Union[plt.Figure, go.Figure]:
        if use_plotly:
            fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.03)
            for i, column in enumerate(self.columns):
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df[column], mode='lines', name=column,
                               line=dict(color=line_color, width=line_width), opacity=opacity),
                    row=i + 1, col=1)
                fig.update_yaxes(title_text=column, row=i + 1, col=1)
            fig.update_layout(height=figsize[1] * 100, width=figsize[0] * 100, showlegend=False)
            return fig
        else:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(7, 1, figure=fig, hspace=0.3)
            t_min, t_max = df['timestamp'].min(), df['timestamp'].max()
            for i, column in enumerate(self.columns):
                ax = fig.add_subplot(gs[i, 0])
                ax.plot(df['timestamp'], df[column], linewidth=line_width, color=line_color, alpha=opacity)
                ax.set_ylabel(column)
                if i < len(self.columns) - 1:
                    ax.tick_params(axis='x', labelsize=0)
                else:
                    ax.set_xlabel('timestamp')
                ax.set_xlim(t_min, t_max)
            return fig

