"""
This app aims to provide an interactive way of demonstrating the AI

Author: Yaolin Ge
Date: 2024-11-28
"""
import streamlit as st
from Components.SupervisedLearningPage import SupervisedLearningPage


st.set_page_config(
    page_title="ML Playground",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yaolinge.github.io/',
        'Report a bug': "https://yaolinge.github.io/",
        'About': "This is a machine learning playground app. Have fun!"
    }
)

selected_page = st.sidebar.selectbox("Select a page", ["Supervised Learning",
                                                       "Unsupervised Learning", 
                                                       "Neural Networks", 
                                                       "Generative AI"], index=0)

if selected_page == "Supervised Learning":
    SupervisedLearningPage().render()
elif selected_page == "Unsupervised Learning":
    st.write("Unsupervised Learning page coming soon!")
elif selected_page == "Neural Networks":
    st.write("Neural Networks page coming soon!")
elif selected_page == "Generative AI":
    st.write("Generative AI page coming soon!")
else:
    st.write("Invalid page selection")

