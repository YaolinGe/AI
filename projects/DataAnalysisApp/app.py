"""
This app aims to provide an interactive way of building an inernal data analysis tool

Author: Yaolin Ge
Date: 2024-10-17
"""
import streamlit as st
import Components.SegmenterAnalysisPage as SegmenterAnalysisPage
import Components.BatchAnalyzerPage as StatisticalReferencePage
import Components.LLMAgentPage as LLMAgentPage
import Components.IncutDetectionPage as IncutDetectionPage
from Components.DataAnnotationPage import DataAnnotationPage


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

selected_page = st.sidebar.selectbox("Select a page", ["Data Annotation", 
                                                       "Incut Detection", 
                                                       "LLM Agent", 
                                                       "Statistical Reference Builder", 
                                                       "Segmenter Analysis"], index=0)

if selected_page == "Segmenter Analysis":
    SegmenterAnalysisPage.renderPage()
elif selected_page == "Statistical Reference Builder":
    StatisticalReferencePage.renderPage()
elif selected_page == "LLM Agent":
    LLMAgentPage.renderPage()
elif selected_page == "Incut Detection":
    IncutDetectionPage.renderPage()
elif selected_page == "Data Annotation":
    # DataAnnotationPage.renderPage()
    DataAnnotationPage().render()
else:
    st.write("Invalid page selection")

