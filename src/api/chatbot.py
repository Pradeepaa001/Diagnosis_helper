import google.generativeai as genai
import os
from fastapi import FastAPI

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

@app.get("/chatbot")
def chatbot_query(query: str):
    response = genai.generate(model="gemini-pro", prompt=query)
    return {"response": response.text}
