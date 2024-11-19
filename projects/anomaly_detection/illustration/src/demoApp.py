"""
This will be the streamlit app for the anomaly detection illustration.

Author: Yaolin Ge
Date: 2024-08-30
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Signal import Signal
from DataHandler import DataHandler



st.set_page_config(
    page_title="Anomaly Detection Illustration",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yaolinge.github.io/',
        'Report a bug': "https://yaolinge.github.io/",
        'About': "This is a demo app for anomaly detection illustration. Have fun!"
    }
)

st.sidebar.title('Machine learning parameters')
with st.sidebar.expander('Signal parameters'): 
    frequency = st.number_input('Frequency', value=0.15, step=0.01)
    amplitude = st.number_input('Amplitude', value=1.0, step=0.1)
    phase = st.number_input('Phase', value=0.0, step=0.1)
    timestamp = np.arange(0, 20, .1)
    noise = st.toggle('Add noise', value=False)
    if noise: 
        noise_level = st.slider('Noise level', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        signal = Signal(frequency, amplitude, phase, timestamp, noise_level)
    else:
        signal = Signal(frequency, amplitude, phase, timestamp, noise_level=.0)
    signal.generate_signal()

with st.sidebar.expander('Model parameters'): 
    look_back = st.number_input('Look back', value=10)
    look_forward = st.number_input('Look forward', value=1)
    isAutoEncoder = st.toggle('AutoEncoder', value=False)
    data_handler = DataHandler(look_back, look_forward, signal, isAutoEncoder=isAutoEncoder)
    data_handler.create_dataset()
    



st.title('Anomaly Detection Illustration')
fig = signal.display()
st.plotly_chart(fig)

