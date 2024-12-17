import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Union, Optional, List, Dict
import warnings

warnings.filterwarnings("ignore")


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
                 text_color: str = "black",
                 incut: bool = False) -> Union[plt.Figure, go.Figure]:
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
            incut: Whether to highlight incut regions
        """
        if exclude_cols is None:
            exclude_cols = []

        timestamp_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if timestamp_col is None:
            df['timestamp'] = np.arange(len(df))
            timestamp_col = 'timestamp'

        plot_cols = [col for col in df.columns if col != timestamp_col and col not in exclude_cols]

        if not plot_cols:
            raise ValueError("No columns to plot after excluding timestamp and specified columns.")

        # Check if 'incut' column exists and is boolean
        has_incut = 'incut' in df.columns and df['incut'].dtype == bool

        n_plots = len(plot_cols)

        if line_color is None:
            colors = self.default_colors
        elif isinstance(line_color, str):
            colors = [line_color] * n_plots
        else:
            colors = line_color

        # Function to find continuous True regions
        def find_incut_regions(incut_series):
            incut_changes = incut_series.ne(incut_series.shift()).cumsum()
            return incut_series.groupby(incut_changes).apply(
                lambda x: (x.index[0], x.index[-1]) if x.iloc[0] else None
            ).dropna()

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

                # Add incut regions for Plotly
                y_min, y_max = df[column].min(), df[column].max()
                y_range = y_max - y_min
                y_padding = y_range * 0.05  # 5% padding
                if incut and has_incut:
                    incut_regions = find_incut_regions(df['incut'])
                    for start, end in incut_regions:
                        fig.add_shape(
                            type="rect",
                            x0=df[timestamp_col].iloc[start],
                            x1=df[timestamp_col].iloc[end],
                            y0=y_min - y_padding,
                            y1=y_max + y_padding,
                            fillcolor=f"rgba(255, 0, 0, 0.25)",
                            line=dict(width=0),
                            layer="below",
                            row=idx + 1,
                            col=1
                        )
                        # fig.add_shape(
                        #     type='rect',
                        #     x0=df[timestamp_col].iloc[start],
                        #     x1=df[timestamp_col].iloc[end],
                        #     y0=0,
                        #     y1=1,
                        #     yref=f'y{idx + 1} domain',
                        #     fillcolor='red',
                        #     opacity=0.2,
                        #     layer='below',
                        #     line_width=0,
                        #     row=idx + 1,
                        #     col=1
                        # )

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
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
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

                # Add incut regions for Matplotlib
                if incut and has_incut:
                    incut_regions = find_incut_regions(df['incut'])
                    for start, end in incut_regions:
                        ax.axvspan(
                            df[timestamp_col].iloc[start],
                            df[timestamp_col].iloc[end],
                            color='red',
                            alpha=0.2
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

    def plot_segmented(self,
                       segmented_df: Dict[str, pd.DataFrame],
                       original_df: Optional[pd.DataFrame] = None,
                       line_color: Union[str, List[str]] = None,
                       segment_colors: Optional[List[str]] = None,
                       original_color: str = "#333333",
                       original_style: str = "dots",  # "line", "dots", or "both"
                       line_width: float = 1.0,
                       height_per_plot: float = 90,
                       plot_width: int = 1500,
                       use_plotly: bool = False,
                       opacity: float = 1.0,
                       exclude_cols: Optional[List[str]] = None,
                       title: str = "Segmented Time Series",
                       text_color: str = "black",
                       show_segment_labels: bool = False,
                       boundary_color: str = "gray",
                       boundary_style: str = "--",
                       boundary_width: float = 1.0) -> Union[plt.Figure, go.Figure]:
        """
        Create vertically stacked line plots for segmented time series data with different colors for each segment.

        Args:
            segmented_df: Dictionary of segmented DataFrames where keys are segment names
            original_df: Optional original DataFrame before segmentation
            line_color: Color(s) for the signal lines. Can be single color or list of colors
            segment_colors: Optional list of colors for different segments
            original_color: Color for the original data
            original_style: Style for original data ("line", "dots", or "both")
            line_width: Width of the lines
            height_per_plot: Height of each subplot in pixels
            plot_width: Total width of the plot in pixels
            use_plotly: Whether to use Plotly instead of Matplotlib
            opacity: Opacity of the signal lines
            exclude_cols: List of column names to exclude from plotting
            title: Title for the entire figure
            text_color: Color for text elements
            show_segment_labels: Whether to show segment labels on the plot
            boundary_color: Color for segment boundary lines
            boundary_style: Style for boundary lines ("--", "-.", ":", etc.)
            boundary_width: Width of boundary lines

        Returns:
            Matplotlib or Plotly figure object
        """
        if exclude_cols is None:
            exclude_cols = []

        # Get the first DataFrame to determine structure
        first_df = next(iter(segmented_df.values()))
        timestamp_col = first_df.columns[0]
        plot_cols = [col for col in first_df.columns if col != timestamp_col and col not in exclude_cols]

        if not plot_cols:
            raise ValueError("No columns to plot after excluding timestamp and specified columns.")

        n_plots = len(plot_cols)
        n_segments = len(segmented_df)

        # Prepare colors
        if line_color is None:
            colors = self.default_colors
        elif isinstance(line_color, str):
            colors = [line_color] * n_plots
        else:
            colors = line_color[:n_plots]

        if segment_colors is None:
            import colorsys
            # Generate distinct colors for segments
            segment_colors = [
                '#' + ''.join([hex(int(x * 255))[2:].zfill(2) for x in colorsys.hsv_to_rgb(i / n_segments, 0.7, 0.9)])
                for i in range(n_segments)
            ]

        # Collect all segment boundaries
        boundaries = []
        for seg_df in segmented_df.values():
            boundaries.extend([seg_df[timestamp_col].iloc[0], seg_df[timestamp_col].iloc[-1]])
        boundaries = sorted(list(set(boundaries)))  # Remove duplicates and sort

        if use_plotly:
            fig = make_subplots(
                rows=n_plots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=plot_cols
            )

            for idx, column in enumerate(plot_cols):
                # Plot original data if provided
                if original_df is not None:
                    if original_style in ["dots", "both"]:
                        fig.add_trace(
                            go.Scatter(
                                x=original_df[timestamp_col],
                                y=original_df[column],
                                mode='markers',
                                name=f"{column} (original)",
                                marker=dict(color=original_color, size=3),
                                opacity=opacity * 0.5,
                                showlegend=(idx == 0)
                            ),
                            row=idx + 1,
                            col=1
                        )
                    if original_style in ["line", "both"]:
                        fig.add_trace(
                            go.Scatter(
                                x=original_df[timestamp_col],
                                y=original_df[column],
                                mode='lines',
                                name=f"{column} (original)",
                                line=dict(color=original_color, width=line_width / 2),
                                opacity=opacity * 0.3,
                                showlegend=False
                            ),
                            row=idx + 1,
                            col=1
                        )

                # Plot each segment
                for seg_idx, (seg_name, seg_df) in enumerate(segmented_df.items()):
                    fig.add_trace(
                        go.Scatter(
                            x=seg_df[timestamp_col],
                            y=seg_df[column],
                            mode='lines',
                            name=f"{column} - {seg_name}" if show_segment_labels else column,
                            line=dict(
                                color=segment_colors[seg_idx % len(segment_colors)],
                                width=line_width
                            ),
                            opacity=opacity,
                            showlegend=(idx == 0 and show_segment_labels)
                        ),
                        row=idx + 1,
                        col=1
                    )

                # Add vertical lines at all boundaries
                for boundary in boundaries:
                    fig.add_vline(
                        x=boundary,
                        line_dash="dash",
                        line_color=boundary_color,
                        line_width=boundary_width,
                        opacity=0.5,
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
                title_text=title,
                title_x=0.5,
                title_font=dict(color=text_color),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=80, r=20, t=40, b=40),
                showlegend=show_segment_labels
            )

        else:
            fig_height = height_per_plot * n_plots / 100
            fig_width = plot_width / 100
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = GridSpec(n_plots, 1, figure=fig, hspace=0.15)

            fig.patch.set_alpha(0.0)
            for idx, column in enumerate(plot_cols):
                ax = fig.add_subplot(gs[idx, 0])
                ax.patch.set_alpha(0.0)

                # Plot original data if provided
                if original_df is not None:
                    if original_style in ["dots", "both"]:
                        ax.scatter(
                            original_df[timestamp_col],
                            original_df[column],
                            color=original_color,
                            s=10,
                            alpha=opacity * 0.5,
                            label=f"{column} (original)"
                        )
                    if original_style in ["line", "both"]:
                        ax.plot(
                            original_df[timestamp_col],
                            original_df[column],
                            color=original_color,
                            linewidth=line_width / 2,
                            alpha=opacity * 0.3
                        )

                # Plot each segment
                for seg_idx, (seg_name, seg_df) in enumerate(segmented_df.items()):
                    label = f"{column} - {seg_name}" if show_segment_labels else None
                    ax.plot(
                        seg_df[timestamp_col],
                        seg_df[column],
                        linewidth=line_width,
                        color=segment_colors[seg_idx % len(segment_colors)],
                        alpha=opacity,
                        label=label
                    )

                # Add vertical lines at all boundaries
                for boundary in boundaries:
                    ax.axvline(
                        x=boundary,
                        color=boundary_color,
                        linestyle=boundary_style,
                        linewidth=boundary_width,
                        alpha=0.5
                    )

                ax.set_ylabel(column, color=text_color)
                ax.tick_params(axis='both', colors=text_color)

                if show_segment_labels and idx == 0:
                    ax.legend(loc='upper right')

                if idx == n_plots - 1:
                    ax.set_xlabel(timestamp_col, color=text_color)
                    plt.xticks(rotation=45)

            if title:
                fig.suptitle(title, fontsize=12, color=text_color)

            plt.tight_layout()

        return fig

    def plot_batch_confidence_interval(self,
                                       segment_data_input: dict,
                                       std_scaler: float = 1.96,
                                       line_color: Union[str, List[str]] = None,
                                       line_width: float = 1.5,
                                       height_per_plot: float = 90,
                                       plot_width: int = 1500,
                                       use_plotly: bool = False,
                                       opacity: float = 1.0,
                                       title: str = "Segmented Time Series with Statistical Reference",
                                       text_color: str = "black",
                                       sync: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create vertically stacked line plots for segmented time series data with average and standard deviation.

        Args:
            segment_data_input (dict): Dictionary containing 'data', 'average', and 'std' DataFrames for the segment.
            line_color (Union[str, List[str]]): Color(s) for the average lines
            line_width (float): Width of the lines
            height_per_plot (float): Height of each subplot in pixels
            plot_width (int): Width of the plot in pixels
            use_plotly (bool): Whether to use Plotly instead of Matplotlib
            opacity (float): Opacity of the lines and fill area
            title (str): Title for the plot
            text_color (str): Color for text elements
            sync (bool): Whether to synchronize the x-axis for all subplots
        """
        segment_data = segment_data_input.copy()
        if sync:
            # make the timestamp to be range from 0 to len of each dataframes
            for i in range(len(segment_data['data'])):
                segment_data['data'][i]['timestamp'] = np.arange(len(segment_data['data'][i]))
                segment_data['average']['timestamp'] = np.arange(len(segment_data['average']))
                segment_data['std']['timestamp'] = np.arange(len(segment_data['std']))
        timestamp_col = segment_data['data'][0].columns[0]
        plot_cols = [col for col in segment_data['data'][0].columns if col != timestamp_col]
        n_plots = len(plot_cols)

        if line_color is None:
            colors = self.default_colors
        elif isinstance(line_color, str):
            colors = [line_color] * n_plots
        else:
            colors = line_color[:n_plots]

        if use_plotly:
            fig = make_subplots(
                rows=n_plots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=plot_cols
            )

            for idx, col in enumerate(plot_cols):
                # add all the data traces
                for df in segment_data['data']:
                    fig.add_trace(
                        go.Scatter(
                            x=df[timestamp_col],
                            y=df[col],
                            mode='lines',
                            line=dict(color='lightgray', width=1),
                            showlegend=False
                        ),
                        row=idx + 1,
                        col=1
                    )

                # Add average line
                fig.add_trace(
                    go.Scatter(
                        x=segment_data['average']['timestamp'],
                        y=segment_data['average'][col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors[idx], width=line_width),
                        opacity=opacity
                    ),
                    row=idx + 1,
                    col=1
                )

                # Add confidence bounds
                fig.add_trace(
                    go.Scatter(
                        x=segment_data['average']['timestamp'],
                        y=segment_data['average'][col] + segment_data['std'][col] * std_scaler,
                        mode='lines',
                        line=dict(color=colors[idx], width=line_width / 2),
                        fillcolor=f"rgba(0, 255, 0, 0.25)",
                        fill=None,
                        showlegend=False,
                        opacity=opacity
                    ),
                    row=idx + 1,
                    col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=segment_data['average']['timestamp'],
                        y=segment_data['average'][col] - segment_data['std'][col] * std_scaler,
                        mode='lines',
                        line=dict(color=colors[idx], width=line_width / 2),
                        fillcolor=f"rgba(0, 255, 0, 0.25)",
                        fill='tonexty',
                        showlegend=False,
                        opacity=opacity
                    ),
                    row=idx + 1,
                    col=1
                )

                fig.update_yaxes(
                    title_text=col,
                    title_font=dict(color=text_color),
                    tickfont=dict(color=text_color),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
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
                title_text=title,
                title_x=0.5,
                showlegend=False,
                title_font=dict(color=text_color),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=80, r=20, t=40, b=40)
            )
        else:
            fig_height = height_per_plot * n_plots / 100
            fig_width = plot_width / 100
            fig, ax = plt.subplots(nrows=n_plots, ncols=1, figsize=(fig_width, fig_height), sharex=True)
            fig.patch.set_alpha(0.0)

            for idx, col in enumerate(plot_cols):
                ax[idx].patch.set_alpha(0.0)
                # Plot all data traces
                for df in segment_data['data']:
                    ax[idx].plot(df[timestamp_col],
                                 df[col],
                                 linewidth=1,
                                 color='lightgray',
                                 alpha=0.5)

                # Plot average line
                ax[idx].plot(segment_data['average']['timestamp'],
                             segment_data['average'][col],
                             linewidth=line_width,
                             color=colors[idx],
                             label=col)

                # Plot confidence bounds
                ax[idx].fill_between(segment_data['average']['timestamp'],
                                     segment_data['average'][col] - segment_data['std'][col] * std_scaler,
                                     segment_data['average'][col] + segment_data['std'][col] * std_scaler,
                                     color="green",
                                     alpha=opacity * 0.3)

                ax[idx].set_ylabel(col, color=text_color)
                ax[idx].tick_params(axis='both', colors=text_color)
                ax[idx].grid(color='lightgray', linestyle='--', linewidth=0.5)

                if idx == n_plots - 1:
                    ax[idx].set_xlabel(timestamp_col, color=text_color)
                    plt.xticks(rotation=45)

            if title:
                fig.suptitle(title, fontsize=12, color=text_color)

            plt.tight_layout()

        return fig

    def lineplot_with_poi(self,
                    df: pd.DataFrame,
                    poi: pd.DataFrame, 
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
                poi: Timestamps to add Point of Interests.
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
                    
                    for _, row in poi.iterrows():
                        fig.add_shape(
                            type="rect",
                            x0=row['InCutTime'],
                            x1=row['OutOfCutTime'],
                            y0=0,
                            y1=1,
                            fillcolor="rgba(255, 0, 0, 0.2)",
                            line=dict(width=0),
                            layer="below",
                            xref=f"x{idx + 1}",
                            yref="paper"
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

                    for _, row in poi.iterrows():
                        ax.axvspan(row['InCutTime'], row['OutOfCutTime'], color='red', alpha=0.2)

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

    def lineplot_with_rect(self,
                    df: pd.DataFrame,
                    t_start: float, 
                    t_end: float,
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
                t_start: Start time of the rectangle
                t_end: End time of the rectangle
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
                    
                    y_min = df[column].min()
                    y_max = df[column].max()
                    fig.add_shape(
                        type="rect",
                        x0=t_start,
                        x1=t_end,
                        y0=y_min,
                        y1=y_max,
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        line=dict(width=0),
                        layer="below",
                        xref=f"x{idx + 1}",
                        yref=f"y{idx + 1}"
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
                fig.update_xaxes(range=[t_start, t_end])

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
                    
                    ax.axvspan(t_start, t_end, color='red', alpha=0.2)

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
                    ax.set_xlim(t_start, t_end)

                if title:
                    fig.suptitle(title, fontsize=12, color=text_color)

                plt.tight_layout()

            return fig
    