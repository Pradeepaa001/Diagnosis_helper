import streamlit as st
import requests

st.title("AI Diagnostic Assistant")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Diagnosis", "Medical Chatbot"])

if page == "Diagnosis":
    st.header("Upload an X-ray/MRI for Diagnosis")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/diagnose", files=files)
        st.write("Result:", response.json()["prediction"])

elif page == "Medical Chatbot":
    st.header("Ask a medical question")
    user_input = st.text_input("Your query:")
    
    if st.button("Ask"):
        response = requests.get(f"http://127.0.0.1:8000/chatbot?query={user_input}")
        st.write("Response:", response.json()["response"])
