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
                height_per_plot: float = 90,  # Height in pixels for each subplot
                plot_width: int = 1500,  # Total width in pixels
                use_plotly: bool = False,
                opacity: float = 1.0,
                exclude_cols: List[str] = None,
                title: str = "",
                text_color: str = "black") -> Union[plt.Figure, go.Figure]:
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
            text_color: Color for text elements like axis labels, ticks, and title
        """
        if exclude_cols is None:
            exclude_cols = []

        timestamp_col = df.columns[0]
        plot_cols = [col for col in df.columns if col != timestamp_col and col not in exclude_cols]

        if not plot_cols:
            raise ValueError("No columns to plot after excluding timestamp and specified columns.")

        n_plots = len(plot_cols)

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
                vertical_spacing=0.01,
                subplot_titles=plot_cols
            )

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
                fig.update_yaxes(
                    title_text=column,
                    title_font=dict(color=text_color),
                    tickfont=dict(color=text_color),
                    row=idx + 1,
                    col=1,
                )
                if idx == n_plots - 1:
                    fig.update_xaxes(
                        title_text=timestamp_col,
                        title_font=dict(color=text_color),
                        tickfont=dict(color=text_color),
                        row=idx + 1,
                        col=1
                    )
            
            fig.update_layout(
                height=height_per_plot * n_plots,
                width=plot_width,
                showlegend=False,
                title_text=title,
                title_x=0.5,
                title_font=dict(color=text_color),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

        else:
            fig_height = height_per_plot * n_plots / 100
            fig_width = plot_width / 100
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = GridSpec(n_plots, 1, figure=fig, hspace=0.01)

            fig.patch.set_alpha(0.0)
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

                ax.patch.set_alpha(0.0)
                ax.set_ylabel(f"{column}", color=text_color)
                ax.tick_params(axis='y', colors=text_color)
                ax.set_xlim(t_min, t_max)

                if idx < n_plots - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(timestamp_col, color=text_color)
                    plt.xticks(rotation=45, color=text_color)
                ax.tick_params(axis='x', colors=text_color)

            if title:
                fig.suptitle(title, fontsize=12, color=text_color)

            plt.tight_layout()

        return fig

    def segmentplot(self,
                    df: pd.DataFrame,
                    segments: List[int],
                    line_color: Union[str, List[str]] = None,
                    segment_color: str = 'red',
                    line_width: float = 1.0,
                    height_per_plot: float = 90,
                    plot_width: int = 1500,
                    use_plotly: bool = False,
                    opacity: float = 1.0,
                    exclude_cols: List[str] = None,
                    title: str = "",
                    text_color: str = "black") -> Union[plt.Figure, go.Figure]:
        """
        Plot time series data with highlighted segments.

        Args:
            df: Input DataFrame where first column is timestamp and others are signals
            segments: List[int]: breakpoints as indices
            line_color: Color(s) for the signal lines
            segment_color: Color for the highlighted segments
            line_width: Width of the lines
            height_per_plot: Height of each subplot in pixels
            plot_width: Total width of the plot in pixels
            use_plotly: Whether to use Plotly instead of Matplotlib
            opacity: Opacity of the signal lines
            exclude_cols: List of column names to exclude from plotting
            title: Title for the entire figure
            text_color: Color for text elements
        """
        if exclude_cols is None:
            exclude_cols = []

        timestamp_col = df.columns[0]
        plot_cols = [col for col in df.columns if col != timestamp_col and col not in exclude_cols]

        if not plot_cols:
            raise ValueError("No columns to plot after excluding timestamp and specified columns.")

        n_plots = len(plot_cols)

        if line_color is None:
            colors = self.default_colors
        elif isinstance(line_color, str):
            colors = [line_color] * n_plots
        else:
            colors = line_color[:n_plots]

        highlight_pairs = []
        sorted_bkps = sorted([0] + list(segments) + [len(df) - 1])
        highlight_pairs = [(sorted_bkps[i], sorted_bkps[i + 1])
                           for i in range(1, len(sorted_bkps) - 1, 2)]

        if use_plotly:
            fig = make_subplots(
                rows=n_plots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=plot_cols
            )

            for idx, column in enumerate(plot_cols):
                # Add signal trace
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

                # Add highlighted segments
                y_min, y_max = df[column].min(), df[column].max()
                y_range = y_max - y_min
                y_padding = y_range * 0.05  # 5% padding

                for start, end in highlight_pairs:
                    fig.add_shape(
                        type="rect",
                        x0=df[timestamp_col].iloc[start],
                        x1=df[timestamp_col].iloc[end],
                        y0=y_min - y_padding,
                        y1=y_max + y_padding,
                        fillcolor=f"rgba(255, 0, 0, 0.25)",
                        line=dict(width=0),
                        layer="below",
                        xref=f"x{idx + 1}",
                        yref=f"y{idx + 1}"
                    )

                # Update axes for each subplot
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=False,
                    row=idx + 1,
                    col=1
                )
                fig.update_yaxes(
                    title_text=column,
                    title_font=dict(color=text_color),
                    tickfont=dict(color=text_color),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=False,
                    row=idx + 1,
                    col=1
                )

                if idx == n_plots - 1:
                    fig.update_xaxes(
                        title_text=timestamp_col,
                        title_font=dict(color=text_color),
                        tickfont=dict(color=text_color),
                        row=idx + 1,
                        col=1
                    )

            fig.update_layout(
                height=height_per_plot * n_plots,
                width=plot_width,
                showlegend=False,
                title_text=title,
                title_x=0.5,
                title_font=dict(color=text_color),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=80, r=20, t=40, b=40)
            )
        else:
            fig_height = height_per_plot * n_plots / 100
            fig_width = plot_width / 100
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = GridSpec(n_plots, 1, figure=fig, hspace=0.15)

            fig.patch.set_alpha(0.0)
            for idx, column in enumerate(plot_cols):
                ax = fig.add_subplot(gs[idx, 0])

                # Plot highlighted segments
                for start, end in highlight_pairs:
                    ax.axvspan(df[timestamp_col].loc[start],
                               df[timestamp_col].iloc[end],
                               color=segment_color,
                               alpha=0.2)
                ax.patch.set_alpha(0.0)
                # Plot signal
                ax.plot(
                    df[timestamp_col],
                    df[column],
                    linewidth=line_width,
                    color=colors[idx % len(colors)],
                    alpha=opacity,
                    label=column
                )

                # Add vertical lines at segment boundaries
                for start, end in highlight_pairs:
                    ax.axvline(x=df[timestamp_col].iloc[start],
                               color='red',
                               linestyle='--',
                               alpha=0.5)
                    ax.axvline(x=df[timestamp_col].iloc[end],
                               color='red',
                               linestyle='--',
                               alpha=0.5)

                ax.set_ylabel(column, color=text_color)
                ax.tick_params(axis='both', colors=text_color)
                # ax.legend(loc='upper right')

                if idx == n_plots - 1:
                    ax.set_xlabel(timestamp_col, color=text_color)
                    plt.xticks(rotation=45)

            if title:
                fig.suptitle(title, fontsize=12, color=text_color)

            plt.tight_layout()

        return fig