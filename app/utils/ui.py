import streamlit as st
import base64

def set_background(image_file: str):
    """Sets a background image as Base64-encoded CSS."""
    with open(image_file, "rb") as f:
        img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()

    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
