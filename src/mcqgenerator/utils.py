import pandas as pd
import PyPDF2
import streamlit as st

import pdfplumber

def read_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        st.error("Unsupported file type! Please upload a TXT or PDF file.")
        return None


def get_table_data(quiz_data: dict):
    """
    Converts a quiz dictionary into a list of dicts suitable for pd.DataFrame.
    Each dict contains Question, Options A-D, and Answer.
    """
    table_data = []
    for q in quiz_data.get("questions", []):
        options = q.get("options", [])
        table_data.append({
            "Question": q.get("question", ""),
            "Option A": options[0] if len(options) > 0 else "",
            "Option B": options[1] if len(options) > 1 else "",
            "Option C": options[2] if len(options) > 2 else "",
            "Option D": options[3] if len(options) > 3 else "",
            "Answer": q.get("answer", "")
        })
    return table_data
