"""
Data Annotation Page for the Data Analysis App.

Improvements:
- State management using st.session_state
- Zoom functionality
- Annotation highlighting
- Performance optimizations

Created on 2024-11-18
Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from Visualizer import Visualizer
from DataAnnotator import DataAnnotator
from InCutDetector import InCutDetector
from Logger import Logger

class DataAnnotationPage:
    def __init__(self):
        # Initialize visualizer and data annotator
        self.visualizer = Visualizer()
        self.dataAnnotator = DataAnnotator(cache_folder=".cache", autosave_interval=10.0)
        self.incutDetector = InCutDetector()
        self.logger = Logger()
        
        # Initialize session state if not exists
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize or reset session state variables."""
        defaults = {
            'data_source': 'Missy',
            'filename': None,
            'df': None,
            't_start': None,
            't_end': None,
            'usePlotly': False,
            'annotation_filepath': None, 
            'window_size': 20,
            'incut_detector': self.incutDetector,
            'visualizer': self.visualizer,
            'dataAnnotator': self.dataAnnotator
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _load_data(self, data_source):
        """Load data based on the selected data source."""
        if data_source == 'Missy':
            folderpath = r'datasets'
            filenames = [f for f in os.listdir(folderpath) 
                         if f.endswith('.csv') and 
                         "Anomaly" not in f and 
                         "POI" not in f]

            filename = st.sidebar.selectbox('Select a file', filenames, 
                                            index=filenames.index(st.session_state.get('filename', filenames[0])) 
                                            if st.session_state.get('filename') in filenames else 0)

            st.session_state.filename = filename
            data_path = os.path.join(folderpath, filename)
            annotation_path = os.path.join("annotations", filename.replace(".csv", "_annotation.csv"))

            # Load DataFrame
            df = pd.read_csv(data_path)
            st.session_state.df = df
            st.session_state.annotation_filepath = annotation_path
            st.session_state.t_start = df['timestamp'].values[0]
            st.session_state.t_end = df['timestamp'].values[-1]
            # st.session_state.t_start = 0 
            # st.session_state.t_end = 400
            st.session_state.incut_detector.process_incut(st.session_state.df, window_size=20)
        else:
            st.warning('Other data source is not implemented yet!')

    def render(self):
        """Main rendering method for the Streamlit app."""
        st.sidebar.title('Parameters')

        # Data source selection
        st.session_state.data_source = st.sidebar.radio(
            'Data source', 
            ['Missy', 'other', 'GulBox'], 
            index=['Missy', 'other'].index(st.session_state.data_source),
            horizontal=True
        )

        # Plot type selection
        st.session_state.usePlotly = st.sidebar.toggle("Use Plotly", st.session_state.usePlotly)

        # Window size 
        st.session_state.window_size = st.sidebar.number_input(
            'Window size', 1, 100, 20
        )

        # Load data
        self._load_data(st.session_state.data_source)

        # Data Annotation Section
        st.sidebar.title('Data Annotation')

        st.session_state.t_start = st.sidebar.number_input(
            'Start time',
            value=float(st.session_state.t_start)
        )
        st.session_state.t_end = st.sidebar.number_input(
            'End time',
            value=float(st.session_state.t_end)
        )
        label = st.sidebar.radio('Label', ['Normal', 'Anomaly', 'Incut'], index=0, horizontal=True)
        if label == "Anomaly": 
            anomaly_description = st.sidebar.text_input('Anomaly Description', 'Anomaly Description')
            label += f": {anomaly_description}"
        save = st.sidebar.button('Save Annotation')

        if save and st.session_state.annotation_filepath:
            self.dataAnnotator.add_annotation(
                st.session_state.annotation_filepath, 
                st.session_state.t_start,
                st.session_state.t_end,
                label
            )
            st.success('Annotation saved!')

        # Create and display plot
        st.title('Data Annotation')

        if st.session_state.df is not None:
            try: 
                df_plot = st.session_state.df[(st.session_state.df['timestamp'] >= st.session_state.t_start) & 
                                            (st.session_state.df['timestamp'] <= st.session_state.t_end)]
                df_plot = df_plot.reset_index(drop=True)
                fig = self.visualizer.lineplot(
                    df = df_plot,
                    line_color = "white",
                    text_color = "white",
                    use_plotly = st.session_state.usePlotly,
                    incut=True
                )
                st.plotly_chart(fig) if st.session_state.usePlotly else st.pyplot(fig)
            except Exception as e:
                self.logger.error(f"t_start: {st.session_state.t_start}, t_end: {st.session_state.t_end}")
                self.logger.error(f"Error plotting data: {e}")
                st.error(f"Error plotting data: {e}")



if __name__ == "__main__":
    DataAnnotationPage().render()