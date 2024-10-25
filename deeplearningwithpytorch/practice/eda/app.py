"""
This app aims to provide an interactive way of building a lstm autoencoder model for detecting anomalies in time series data.

Author: Yaolin Ge
Date: 2024-10-17
"""

import streamlit as st
import os
import asyncio
from CutFileHandler import CutFileHandler
from Gen1CSVHandler import Gen1CSVHandler
from Visualizer import Visualizer
from datetime import datetime
from dataclasses import dataclass

cutFileHandler = CutFileHandler()
gen1CSVHandler = Gen1CSVHandler()
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
        filenames = os.listdir(folderpath)
        filenames = [file for file in filenames if file.endswith('.cut')]
        selected_file = st.sidebar.selectbox('Select a file', filenames)
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
        filenames = os.listdir(folderpath)
        filenames = [filename for filename in filenames if filename.endswith('.cut')]
        filenames_cropped = [filename[18:-4] for filename in filenames]
        selected_file = st.sidebar.selectbox('Select a file', filenames_cropped)
        selected_index = filenames_cropped.index(selected_file)
        st.sidebar.write(f"Selected file index: {selected_index}")
        if selected_file is not None:
            filepath = os.path.join(folderpath, f"{filenames[selected_index]}")
            st.write(f"filepath: {filepath}")
            with st.spinner("Processing file..."):
                gen1CSVHandler.process_file(filepath)
                st.toast("Number of rows: " + str(gen1CSVHandler.df_sync.shape[0]))
            fig = visualizer.lineplot(gen1CSVHandler.df_sync, line_color="white", plot_width=1200, height_per_plot=80, line_width=.5, use_plotly=usePlotly, text_color="white")
            if usePlotly:
                st.plotly_chart(fig)
            else:
                st.pyplot(fig)






