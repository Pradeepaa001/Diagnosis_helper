import streamlit as st
import requests

st.title("Medical Image Diagnosis")

uploaded_file = st.file_uploader("Upload an X-ray or MRI", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/diagnose", files=files)
    st.write("Result:", response.json()["prediction"])
