import streamlit as st
import requests

st.title("Medical Chatbot")

query = st.text_input("Ask a medical question:")

if st.button("Ask"):
    response = requests.get(f"http://127.0.0.1:8000/chatbot?query={query}")
    st.write("Response:", response.json()["response"])
