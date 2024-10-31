"""
StatisticalReferencePage renders the page for the statistical confidence intervals

Author: Yaolin Ge
Date: 2024-10-30
"""
import streamlit as st
import os
from Visualizer import Visualizer
from datetime import datetime
from Gen1CutFileHandler import Gen1CutFileHandler
from StatisticalReferenceBuilder import StatisticalReferenceBuilder


gen1_cutfile_handler = Gen1CutFileHandler()
visualizer = Visualizer()
statistical_reference_builder = StatisticalReferenceBuilder()


def parse_file_meaning(filename: str) -> str:
    try:
        date_str = filename[:8]
        time_str = filename[-6:]
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        time_obj = datetime.strptime(time_str, '%H%M%S')
        return f"{date_obj.strftime('%Y-%m-%d')}, {time_obj.strftime('%H:%M:%S')}"
    except ValueError:
        return "Invalid filename format"



def renderPage():
    # === Sidebar global parameters ==============================================
    st.sidebar.title('Parameters')
    usePlotly = st.sidebar.toggle('usePlotly', True)
    useSync = st.sidebar.toggle('useSync', True)
    resolution_ms = st.sidebar.number_input('Resolution (ms)', value=250)


    # === Sidebar local parameters ===============================================
    # file_type = st.sidebar.radio("File Type", [".csv", ".cut"])
    # processed_data_columns = ['load', 'deflection', 'surfacefinish', 'vibration']




    # === Sidebar ====
    data_source = st.sidebar.radio("Data Source", ["Dan", "Other"])
    if data_source == "Dan":
        folderpath = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin"
        filenames = os.listdir(folderpath)
        filenames = [filename for filename in filenames if filename.endswith('.cut')]

        filenames_selected = st.sidebar.multiselect('Select files', filenames, default=filenames[~5])

        filenames_path = [os.path.join(folderpath, filename) for filename, selected in zip(filenames, filenames_selected) if selected]

        # st.write(filenames_path)

        result = statistical_reference_builder.build_statistical_reference(filenames_path, resolution_ms=resolution_ms)

        segment_selected = st.sidebar.selectbox('Select a segment', list(result.keys()))

        fig = visualizer.plot_statistical_reference(result[segment_selected], line_color="black", text_color="white",
                                                    line_width=.5, use_plotly=usePlotly, sync=useSync)
        st.plotly_chart(fig)


