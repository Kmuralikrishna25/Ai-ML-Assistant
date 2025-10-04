# 🤖 Gemini Data Scientist Agent

This project is an **AI-powered Data Scientist Agent** built with **Streamlit**, **LangChain**, and **Google Gemini (Generative AI)**.  

You can upload a **CSV dataset**, and the app will automatically:
- Summarize the dataset  
- Suggest **data cleaning steps**  
- Recommend **encoding & scaling methods**  
- Identify a **target column** and task type (classification/regression)  
- Suggest suitable **ML algorithms** with reasoning  
- Provide a **training plan** (train/test split, cross-validation, metrics)  
- Generate a short **scikit-learn code snippet**  

---

## 🚀 Features
- 📂 Upload any CSV dataset  
- 🧩 Automatic **chunking + summarization** of large datasets  
- 🤖 AI-powered recommendations from **Gemini 2.5 Flash**  
- 🧠 Structured, human-readable analysis report  
- ⚡ Simple UI built with **Streamlit**  

---

## 🛠️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/AI_ML_Assistant.git
cd AI_ML_Assistant

##Requirements :

streamlit
pandas
python-dotenv
langchain
langchain-google-genai
google-generativeai


🔑 API Key Setup

Get a Google Gemini API key from Google AI Studio
.

Create a .env file in your project root:

GEMINI_API_KEY=your_api_key_here