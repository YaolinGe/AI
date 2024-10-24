"""
This app aims to provide an interactive way of building a lstm autoencoder model for detecting anomalies in time series data.

Author: Yaolin Ge
Date: 2024-10-17
"""

import streamlit as st
import os
import asyncio
from CutFileHandler import CutFileHandler
from Visualizer import Visualizer

cutFileHandler = CutFileHandler()
visualizer = Visualizer()
folderpath = r"C:\Users\nq9093\Downloads\JorgensData"
files = os.listdir(folderpath)
files = [file for file in files if file.endswith('.cut')]


st.set_page_config(
    page_title="Data Analysis App",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yaolinge.github.io/',
        'Report a bug': "https://yaolinge.github.io/",
        'About': "This is a data analysis app. Have fun!"
    }
)

selected_file = st.sidebar.selectbox('Select a file', files)

# Title
st.title('🧪')
st.sidebar.title('Parameters')
usePlotly = st.sidebar.toggle('usePlotly', True)

# Main content
if selected_file is not None:
    filepath = os.path.join(folderpath, selected_file)
    # cutFileHandler.process_file(filepath, resolution_ms=100)
    with st.spinner("Processing file..."):
        cutFileHandler.process_file(filepath, resolution_ms=1000)
        df = cutFileHandler.get_synchronized_data()

    # st.write(df.head())
    fig = visualizer.lineplot(df, line_color="black", line_width=.5, use_plotly=usePlotly)
    if usePlotly:
        st.plotly_chart(fig)
    else:
        st.pyplot(fig)
