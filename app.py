import os
import pandas as pd
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Gemini + LangChain setup
# -----------------------------
def create_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )


def chunk_dataset(df: pd.DataFrame, chunk_size: int = 2000, overlap: int = 100):
    csv_text = df.to_csv(index=False)
    doc = Document(page_content=csv_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    docs = splitter.split_documents([doc])
    return docs


def analyze_dataset(df: pd.DataFrame):
    llm = create_gemini_model()
    docs = chunk_dataset(df)

    # Summarize dataset in chunks (map-reduce)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    dataset_summary = summarize_chain.run(docs)

    # Ask Gemini for analysis in natural text (not JSON)
    prompt = f"""
    You are a senior AI Data Scientist.
    Based on the dataset summaries below:

    {dataset_summary}

    Please give me a detailed report with:
    1. Data cleaning steps
    2. Encoding methods for categorical columns
    3. Scaling methods for numeric columns
    4. Possible target column and task type
    5. Recommended algorithms with reasoning
    6. Suggested training plan (train/test split, cross-validation, metrics)
    7. A short sklearn-style code snippet for the best algorithm

    Present the answer in a clear, human-readable format with sections.
    """

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🤖 Gemini Data Scientist Agent", layout="wide")

st.title("🤖 Gemini Data Scientist Agent")
st.write("Upload a dataset and let Gemini suggest cleaning steps, encoders, scalers, algorithms, and training plan.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    with st.spinner("Analyzing dataset with Gemini... ⏳"):
        suggestions = analyze_dataset(df)

    st.subheader("🧠 Gemini’s Recommendations")
    st.markdown(suggestions)
