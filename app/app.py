import sys
import os
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
#   FIX PYTHON PATH (ensures imports work in multipage app)
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------
#   PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="GenAI Financial Prediction App",
    layout="wide",
)

# ---------------------------------------------------------
#   BACKGROUND (Base64 safe for multipage apps)
# ---------------------------------------------------------
from utils.ui import set_background
set_background("assets/stock.png")    

# ---------------------------------------------------------
#   WELCOME PAGE HTML
# ---------------------------------------------------------
components.html(
    """
    <style>
    .overlay {
        background: rgba(0,0,0,0.55);
        padding: 50px;
        border-radius: 18px;
        width: 80%;
        margin: auto;
        margin-top: 50px;
    }
    h1, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>

    <div class="overlay">
        <h1 style="color:white; text-align:center; font-size:55px; font-weight:700;">
            📈 GenAI Financial Prediction Platform
        </h1>

        <h3 style="color:#d6e4ff; text-align:center; font-size:24px;">
            Explore model predictions, analyze price movements, and interact with an AI-powered financial assistant.
        </h3>
    </div>
    """,
    height=400,
)

# ---------------------------------------------------------
#   NAV BUTTONS
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Open Insights Dashboard", use_container_width=True):
        st.switch_page("pages/1_Insights_Dashboard.py")

with col2:
    if st.button("Talk To Our Chatbot", use_container_width=True):
        st.switch_page("pages/2_Talk_To_Our_LLM.py")

with col3:
    if st.button("Access Articles Of The Day", use_container_width=True):
        st.switch_page("pages/3_News_Articles.py")

st.markdown("<br><br><br>", unsafe_allow_html=True)
