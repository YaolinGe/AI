"""
This is the main page for the LLMAgent. It is responsible for rendering the main page of the LLMAgent.

Author: Yaolin Ge
Date: 2024-11-20
"""
import streamlit as st
import os
from Visualizer import Visualizer
from datetime import datetime




def renderPage():
    # === Sidebar global parameters ==============================================
    st.sidebar.title('Parameters')
    usePlotly = st.sidebar.toggle('usePlotly', True)
    useSync = st.sidebar.toggle('useSync', True)
    resolution_ms = st.sidebar.number_input('Resolution (ms)', value=250)



    # === Sidebar ====
    st.title("LLM Agent")


    # === Main content === 


