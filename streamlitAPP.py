import pandas as pd
import json
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

# Load the response JSON
with open("./config/workspace/Response.json", 'r') as file:
    RESPONSE_JSON = json.load(file)

# Streamlit app title
st.title("MCQs Creator Application with LangChain 🦜🔗")

# Form for user input
with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

    # Additional form fields
    mcq_count = st.number_input("Number of MCQs", min_value=1, max_value=50)
    subject = st.text_input("Subject")
    tone = st.selectbox("Quiz Tone", ["Simple", "Moderate", "Hard"])
    button = st.form_submit_button("Create MCQs")

    # Generate MCQs on button click
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading..."):
            try:
                text = read_file(uploaded_file)
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone
                    }
                )

                with get_openai_callback() as cb:
                    st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                    st.write(f"Completion Tokens: {cb.completion_tokens}")
                    st.write(f"Total Cost: {cb.total_cost}")

                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index += 1
                            st.table(df)

            except Exception as e:
                st.error(f"An error occurred: {e}")
