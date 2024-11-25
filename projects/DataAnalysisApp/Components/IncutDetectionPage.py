"""
IncutDetectionPage renders the page for the incut detection analysis

Author: Yaolin Ge
Date: 2024-11-25
"""
import streamlit as st
import os
import pandas as pd
from datetime import datetime
from InCutDetector import InCutDetector
from Visualizer import Visualizer


visualizer = Visualizer()
incutDetector = InCutDetector()



def renderPage():
    # === Sidebar global parameters ==============================================
    st.sidebar.title('Parameters')
    usePlotly = st.sidebar.toggle('usePlotly', False)
    runIncut = st.sidebar.toggle('Run Incut Detection', False)
    if runIncut: 
        window_size = st.sidebar.slider('Window size', 1, 100, 10)
    

    data_source = st.sidebar.radio('Data source', ['Missy', 'other'], index=0, horizontal=True)
    if data_source == 'Missy':
        folderpath = r'datasets'
        filenames = os.listdir(folderpath)
        filenames = [f for f in filenames if f.endswith('.csv')]
        filename = st.sidebar.selectbox('Select a file', filenames)
        data_path = os.path.join(folderpath, filename)
        df = pd.read_csv(data_path)
    else:
        st.warning('Other data source is not implemented yet!')
        # data_path = st.sidebar.text_input('Data path', 'data/other.csv')

    
    # === Main page ==============================================================
    st.title('Incut Detection')
    
    if data_source == 'Missy':
        if runIncut:
            with st.spinner('Processing data...'):
                incutDetector.process_incut(df, window_size)
        fig = visualizer.lineplot(df, use_plotly=usePlotly, incut=runIncut)
        if usePlotly:
            st.plotly_chart(fig)
        else: 
            st.pyplot(fig)
