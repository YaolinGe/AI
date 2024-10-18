"""
This app aims to provide an interactive way of building a lstm autoencoder model for detecting anomalies in time series data.

Author: Yaolin Ge
Date: 2024-10-17
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from DataHandler import DataHandler
from Visualizer import Visualizer
from MachineLearning import MachineLearning

dataHandler = DataHandler()
visualizer = Visualizer()
# machineLearning = MachineLearning()

folder_path = r"C:\Users\nq9093\Downloads\CutFilesToYaolin\CutFilesToYaolin"
files = os.listdir(folder_path)
filenames = [file[:-4] for file in files if file.endswith('.cut')]

st.set_page_config(
    page_title="LSTMAutoencoderApp",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yaolinge.github.io/',
        'Report a bug': "https://yaolinge.github.io/",
        'About': "This is a demo LSTM AutoEncoder app. Have fun!"
    }
)

selected_file = st.sidebar.selectbox('Missy dataset', filenames)

# Title
st.title('Demo LSTM AutoEncoder')
st.sidebar.title('Parameters')
usePlotly = st.sidebar.toggle('usePlotly', False)

# Main content
if selected_file is not None:
    filepath = os.path.join(folder_path, selected_file)
    dataHandler.load_synchronized_data(filepath)

    # === 
    t_min = dataHandler.df_sync['timestamp'].min()
    t_max = dataHandler.df_sync['timestamp'].max()
    t_start = st.sidebar.slider('Start Time', t_min, t_max, 8.0)
    t_end = st.sidebar.slider('End Time', t_min, t_max, 15.0)
    num_epochs = st.sidebar.slider('Number of Epochs', 10, 100, 30)
    train_button = st.sidebar.button('Train Model')
    # === 

    if usePlotly:
        fig = visualizer.plotly_data(dataHandler.df_sync, t_start=t_start, t_end=t_end)
        st.plotly_chart(fig)
    else:
        fig = visualizer.plot_data(dataHandler.df_sync, t_start=t_start, t_end=t_end)
        st.pyplot(fig)

    dataHandler.get_cropped_data(t_start, t_end)
    dataHandler.prepare_training_data(dataHandler.df_sync_cropped)

    fig = visualizer.plot_data(dataHandler.df_sync_cropped)
    st.pyplot(fig)
    # if train_button:
    #     machineLearning.train_model(dataHandler.train_loader, dataHandler.val_loader, dataHandler.test_loader,
    #                                 num_epochs=num_epochs)
    #     fig = visualizer.plot_train_val_losses(machineLearning.train_losses, machineLearning.val_losses)
    #     st.pyplot(fig)

    #     machineLearning.predict(dataHandler.df_sync)
    # else:
    #     st.info('Please click the Train Model button to start training.')


else:
    st.info('Please upload a CSV file to start.')

