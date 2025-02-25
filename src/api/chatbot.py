import google.generativeai as genai
import src.config as config

genai.configure(api_key=config.GOOGLE_API_KEY)

# Query Gemini API
def chat_with_gemini(query):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(query)
    return response.text
