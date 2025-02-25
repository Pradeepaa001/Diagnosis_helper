# **Diagnosis_Aid: AI-Powered Medical Diagnostic Assistant**  

## **Overview**  
Diagnosis_Aid is an AI-powered diagnostic assistant designed to assist healthcare professionals by analyzing medical images and answering medical queries. It leverages **CNN-based image classification**, **Federated Learning**, and **Google Gemini API** for an interactive chatbot.  

### **Key Features**  
- **Medical Image Analysis** – Analyzes X-rays/MRIs for preliminary diagnosis.  
- **AI-Powered Chatbot** – Uses **Google Gemini API** for answering health-related queries.  
- **Federated Learning** – Enables privacy-preserving AI training across hospitals.  
- **User-Friendly Interface** – Deployed as a **Streamlit-based Web App**.  

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/diagnosis_aid.git
cd diagnosis_aid
```

### **2. Set Up the Virtual Environment**  
Install **Poetry** if not already installed:  
```bash
pip install poetry
```
Then, create and activate the environment:  
```bash
poetry install
poetry shell
```

### **3. Set Up API Keys**  
Create a `.env` file in the root directory and add your API keys:  
```
GEMINI_API_KEY=your_google_gemini_api_key
```

---

## **Usage**  

### **1. Running the Application**  
Start the **FastAPI backend**:  
```bash
uvicorn src.api.diagnostic_api:app --host 0.0.0.0 --port 8000
```
Start the **Streamlit frontend**:  
```bash
streamlit run src/frontend/app.py
```

### **2. Uploading Medical Images for Analysis**  
- Open the **Streamlit UI** in the browser.  
- Upload an X-ray/MRI image.  
- Get real-time diagnosis results.  

### **3. Using the Chatbot**  
- Enter medical-related queries in the chat window.  
- The chatbot (powered by Gemini API) provides AI-generated responses.  

---

## **Project Structure**  
```
diagnosis_aid/
│── src/
│   ├── api/
│   │   ├── chatbot.py         # Google Gemini chatbot API
│   │   ├── diagnostic_api.py  # X-ray/MRI diagnostic API
│   │   ├── data_loader.py     # Data preprocessing utilities
│   │   ├── utils.py           # Helper functions
│   ├── models/
│   │   ├── model_loader.py    # Loads CNN & Federated models
│   │   ├── federated_train.py # Federated Learning training script
│   ├── frontend/
│   │   ├── app.py             # Streamlit web app
│   │   ├── chatbot_ui.py      # Chatbot UI
│   ├── config.py              # Configuration file
│── tests/                     # Unit tests
│── pyproject.toml             # Poetry dependency management
│── README.md                  # Documentation
│── .env                       # API keys & secrets
```

---

## **Training the Model**  
### **1. Download the Dataset**  
Use **NIH Chest X-ray** or **RSNA Pneumonia Dataset**:  
```bash
mkdir data
cd data
wget https://nihcc.app.box.com/v/ChestXray-NIHCC
```

### **2. Train the Model**  
```bash
python src/models/federated_train.py
```
- The model trains using **Federated Learning** to simulate privacy-preserving data sharing.  
- A pre-trained model is also available for direct use.  

---

## **Dependencies**  
The following dependencies are required (installed via Poetry):  
- **AI & ML**: `torch`, `torchvision`, `tensorflow`, `scikit-learn`, `opencv-python`, `Pillow`  
- **Backend**: `fastapi`, `uvicorn`, `flask`, `pydantic`  
- **Frontend**: `streamlit`  
- **Others**: `numpy`, `pandas`, `dotenv`, `google-generativeai`  

---

## **Future Enhancements**  
- Extend support for **wound/rash detection**.  
- Enable **real-world federated training** with hospital collaboration.  
- Store **patient history** for personalized diagnosis.  

---

## **Contributors**  
- **Pradeepaa001** – [mpradeepaa2020@gmail.com](mailto:mpradeepaa2020@gmail.com)  

