"""
This will be the streamlit app for the anomaly detection illustration.

Author: Yaolin Ge
Date: 2024-08-30
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


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

# page_options = ['🧪 Sense', '🎲 Plan', '🤖 Act']

# selected_page = st.sidebar.selectbox('Adaptive Sampling System', page_options)

# Download the data
# url = "https://drive.google.com/file/d/1mIGUevAlptjYIOdOjpSiVss2VyW-1qsS/view?usp=sharing"
# if not os.path.exists("interpolator_medium.joblib"):
#     response = requests.get(url)
#     with open("interpolator_medium.joblib", "wb") as file:
#         file.write(response.content)

        # if selected_page == '🧪 Sense':
        #     renderSensePage()
        # elif selected_page == '🎲 Plan':
        #     renderPlanPage()
        # elif selected_page == '🤖 Act':
        #     renderActPage()
st.title('Anomaly Detection Illustration')

st.write('This is a simple illustration of anomaly detection using a simple sinusoidal signal.')

