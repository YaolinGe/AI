"""
SegmenterAnalysisPage renders the page for the segmenter analysis.

Author: Yaolin Ge
Date: 2024-10-30
"""
import streamlit as st
import os
import pandas as pd
from CutFileHandler import CutFileHandler
from Gen1CutFileHandler import Gen1CutFileHandler
from Visualizer import Visualizer
from Segmenter.BreakPointDetector import BreakPointDetector
from datetime import datetime

gen1CutFileHandler = CutFileHandler(is_gen2=False, debug=True)
danCSVFileHandler = Gen1CutFileHandler()
gen2CutFileHandler = CutFileHandler(is_gen2=True)
visualizer = Visualizer()
breakpointDetector = BreakPointDetector()


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
    usePlotly = st.sidebar.toggle('usePlotly', False)
    showSegments = st.sidebar.toggle('Show Segments', True)
    reverseSegments = st.sidebar.toggle('Reverse Segments', False)
    onlyRaw = st.sidebar.toggle('Only Raw', False)


    # === Sidebar local parameters ===============================================
    file_type = st.sidebar.radio("File Type", [".csv", ".cut"], index=1, horizontal=True)
    processed_data_columns = ['load', 'deflection', 'surfacefinish', 'vibration']



    # with st.sidebar.expander("Advanced Parameters"):  # TODO: check why expander is not working
    resolution_ms = st.sidebar.number_input('Resolution (ms)', value=1000)
    model_type = st.sidebar.selectbox('Model Type', ['BottomUp', 'Binseg', 'Pelt', 'Window'], index=2)
    pen = st.sidebar.number_input('Penalty', value=100000)
    model = st.sidebar.selectbox('Cost Function', ['l1', 'l2', 'rbf', 'linear', 'normal', 'ar'])
    jump = st.sidebar.number_input('Jump', value=1)
    min_size = st.sidebar.number_input('Min Size', value=1)



    if file_type == ".cut":
        # === Sidebar ====
        data_source = st.sidebar.radio("Data Source", ["Jørgen", "Gen2", "Other"], index=1)
        if data_source == "Jørgen":
            folderpath = r"C:\Data\JorgensData"
            filenames = os.listdir(folderpath)
            filenames = [file for file in filenames if file.endswith('.cut')]
            selected_file = st.sidebar.selectbox('Select a file', filenames)

        # === Main ====
            if selected_file is not None:
                fig = None
                filepath = os.path.join(folderpath, selected_file)
                with st.spinner("Processing file..."):
                    try:
                        gen1CutFileHandler.process_file(filepath, resolution_ms=resolution_ms)
                        st.toast("Number of rows: " + str(gen1CutFileHandler.get_synchronized_data().shape[0]))

                        df = gen1CutFileHandler.get_synchronized_data()
                        if onlyRaw:
                            df = df.loc[:, ~df.columns.str.contains('|'.join(processed_data_columns), case=False)]

                        # === segment data ===
                        if showSegments:
                            signal = df.iloc[:, 1:].to_numpy()
                            result = breakpointDetector.fit(signal, pen=pen, model_type=model_type, model=model, jump=jump, min_size=min_size)
                            st.write(f"Segments: {result}")
                            st.write(f"Number of df: {len(df)}")

                    except(Exception) as e:
                        st.error(e)

                if showSegments:
                    if len(result) < 100:
                        if reverseSegments:
                            fig = visualizer.segmentplot(df, result[1:], line_color="white", line_width=.5, use_plotly=usePlotly, text_color="white", plot_width=1200, height_per_plot=80)
                        else:
                            fig = visualizer.segmentplot(df, result, line_color="white", line_width=.5, use_plotly=usePlotly, text_color="white", plot_width=1200, height_per_plot=80)
                else:
                    fig = visualizer.lineplot(df, line_color="white", line_width=.5, use_plotly=usePlotly, text_color="white", plot_width=1200, height_per_plot=80)

                if fig:
                    if usePlotly:
                        st.plotly_chart(fig)
                    else:
                        st.pyplot(fig)

        elif data_source == "Gen2":
            missy_datasource = st.sidebar.radio("Missy source", ["1", "2"], index=0, horizontal=True)
            folderpath = rf"C:\Data\MissyDataSet\Missy_Disc{missy_datasource}\Cutfiles"
            filenames = os.listdir(folderpath)
            filenames = [file for file in filenames if file.endswith('.cut')]
            selected_file = st.sidebar.selectbox('Select a file', filenames)
            filepath = os.path.join(folderpath, selected_file)

            with st.spinner("Processing file..."):
                try:
                    gen2CutFileHandler.process_file(filepath, resolution_ms=resolution_ms)
                    df = gen2CutFileHandler.get_synchronized_data()
                    if onlyRaw:
                        df = df.loc[:, ~df.columns.str.contains('|'.join(processed_data_columns), case=False)]

                    # === segment data ===
                    if showSegments:
                        signal = df.iloc[:, 1:].to_numpy()
                        result = breakpointDetector.fit(signal, pen=pen, model_type=model_type, model=model, jump=jump, min_size=min_size)
                        st.write(f"Segments: {result}")
                        st.write(f"Number of df: {len(df)}")

                        if len(result) < 100:
                            if reverseSegments:
                                fig = visualizer.segmentplot(df, result[1:], line_color="white", line_width=.5, use_plotly=usePlotly, text_color="white", plot_width=1200, height_per_plot=80)
                            else:
                                fig = visualizer.segmentplot(df, result, line_color="white", line_width=.5, use_plotly=usePlotly, text_color="white", plot_width=1200, height_per_plot=80)
                    else:
                        fig = visualizer.lineplot(df, line_color="white", line_width=.5, use_plotly=usePlotly, text_color="white", plot_width=1200, height_per_plot=80)
                    
                    if fig:
                        if usePlotly:
                            st.plotly_chart(fig)
                        else:
                            st.pyplot(fig)

                except(Exception) as e:
                    st.error(e)


    elif file_type == ".csv":
        # === Sidebar ====
        data_source = st.sidebar.radio("Data Source", ["Other", "Missy", "Dan"])
        selected_file = None
        if data_source == "Dan":
            folderpath = r"C:\Data\Gen1CutFile"
            filenames = os.listdir(folderpath)
            filenames = [filename for filename in filenames if filename.endswith('.cut')]
            filenames_cropped = [filename[18:-4] for filename in filenames]
            selected_file = st.sidebar.selectbox('Select a file', filenames_cropped)
            selected_index = filenames_cropped.index(selected_file)
        elif data_source == "Missy":
            folderpath = "datasets"
            filenames = os.listdir(folderpath)
            filenames = [filename for filename in filenames if filename.endswith('.csv')]
            selected_file = st.sidebar.selectbox('Select a file', filenames)
            selected_index = filenames.index(selected_file)
        else: 
            st.error("Other data source not implemented yet")


        # === Main ====
        if selected_file is not None:
            fig = None
            filepath = os.path.join(folderpath, f"{filenames[selected_index]}")
            st.write(f"filepath: {filepath}")
            with st.spinner("Processing file..."):
                if data_source == "Dan":
                    danCSVFileHandler.process_file(filepath, resolution_ms=resolution_ms)
                    df = danCSVFileHandler.df_sync
                elif data_source == "Missy":
                    df = pd.read_csv(filepath)

                if onlyRaw:
                    st.write("Remove processed data columns: ", '|'.join(processed_data_columns))
                    df = df.loc[:, ~df.columns.str.contains('|'.join(processed_data_columns), case=False)]

                if showSegments:
                    signal = df.iloc[:, 1:].to_numpy()
                    result = breakpointDetector.fit(signal, pen=pen, model_type=model_type, model=model, jump=jump, min_size=min_size)
                    st.write(f"Segments: {result}")
                    st.write(f"Number of df: {len(df)}")

            if showSegments:
                if len(result) < 100:
                    if reverseSegments:
                        fig = visualizer.segmentplot(df, result[1:], line_color="white", plot_width=1200, height_per_plot=80, line_width=.5, use_plotly=usePlotly, text_color="white")
                    else:
                        fig = visualizer.segmentplot(df, result, line_color="white", plot_width=1200, height_per_plot=80, line_width=.5, use_plotly=usePlotly, text_color="white")
            else:
                fig = visualizer.lineplot(df, line_color="white", plot_width=1200, height_per_plot=80, line_width=.5, use_plotly=usePlotly, text_color="white")

            if fig:
                if usePlotly:
                    st.plotly_chart(fig)
                else:
                    st.pyplot(fig)