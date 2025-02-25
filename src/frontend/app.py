import streamlit as st
from PIL import Image
from src.api.diagnostic_api import diagnose
from src.api.chatbot import chat_with_gemini

st.title("🩺 AI-Powered Medical Assistant")

tab1, tab2 = st.tabs(["🖼️ Diagnosis", "💬 Chatbot"])

# Diagnosis Tab
with tab1:
    uploaded_file = st.file_uploader("Upload MRI/X-ray", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = diagnose(uploaded_file)
        st.write(f"**Diagnosis Result:** {prediction}")

# Chatbot Tab
with tab2:
    user_query = st.text_input("Ask a medical question:")
    if user_query:
        response = chat_with_gemini(user_query)
        st.write(f"**Gemini Response:** {response}")
