"""
This app aims to provide an interactive way of building an inernal data analysis tool

Author: Yaolin Ge
Date: 2024-10-17
"""
import streamlit as st
import Components.SegmenterAnalysisPage as SegmenterAnalysisPage
import Components.BatchAnalyzerPage as StatisticalReferencePage


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

selected_page = st.sidebar.selectbox("Select a page", ["Statistical Reference Builder", "Segmenter Analysis"])

if selected_page == "Segmenter Analysis":
    SegmenterAnalysisPage.renderPage()
else:
    StatisticalReferencePage.renderPage()
