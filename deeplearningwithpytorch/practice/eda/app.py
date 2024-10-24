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
from datetime import datetime
from dataclasses import dataclass

cutFileHandler = CutFileHandler()
visualizer = Visualizer()


def parse_file_meaning(filename: str) -> str:
    try:
        date_str = filename[:8]
        time_str = filename[-6:]
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        time_obj = datetime.strptime(time_str, '%H%M%S')
        return f"{date_obj.strftime('%Y-%m-%d')}, {time_obj.strftime('%H:%M:%S')}"
    except ValueError:
        return "Invalid filename format"


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

# === Sidebar global parameters ==============================================
st.sidebar.title('Parameters')
usePlotly = st.sidebar.toggle('usePlotly', True)


# === Sidebar local parameters ===============================================
file_type = st.sidebar.radio("File Type", [".csv", ".cut"])





if file_type == ".cut":
    # === Sidebar ====
    data_source = st.sidebar.radio("Data Source", ["Jørgen", "Other"])
    if data_source == "Jørgen":
        folderpath = r"C:\Users\nq9093\Downloads\JorgensData"
        files = os.listdir(folderpath)
        files = [file for file in files if file.endswith('.cut')]
        selected_file = st.sidebar.selectbox('Select a file', files)
        resolution_ms = st.sidebar.number_input('Resolution (ms)', value=1000)

    # === Main ====
        if selected_file is not None:
            filepath = os.path.join(folderpath, selected_file)
            with st.spinner("Processing file..."):
                cutFileHandler.process_file(filepath, resolution_ms=resolution_ms)
                st.toast("Number of rows: " + str(cutFileHandler.get_synchronized_data().shape[0]))
            fig = visualizer.lineplot(cutFileHandler.get_synchronized_data(), line_color="white", line_width=.5, use_plotly=usePlotly)
            if usePlotly:
                st.plotly_chart(fig)
            else:
                st.pyplot(fig)

elif file_type == ".csv":
    data_source = st.sidebar.radio("Data Source", ["Dan", "Other"])
    if data_source == "Dan":
        folderpath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin"
        files = os.listdir(folderpath)
        files = [file[18:-4] for file in files if file.endswith('.cut')]
        parsed_files = [parse_file_meaning(file) for file in files]
        selected_file = st.sidebar.selectbox('Select a file', parsed_files)
        st.write(selected_file)





