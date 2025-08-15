import os
import json
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.mcqgenerator.MCQGenerator import build_llm, build_generation_chain, summarize_if_needed, MCQSet
from src.mcqgenerator.utils import read_pdf, read_txt, read_docx, clean_text

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False

st.title("📝 MCQ Generator")

# API Key Check
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment. Please set it in `.env`.")
    st.stop()

# Sidebar Configuration
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Model",
    ["llama3-8b-8192", "llama3-70b-8192"],
    index=0,
    help="Free tier has 6000 tokens/minute limit. Reduce input size or upgrade."
)

# File Upload Section
uploaded = st.file_uploader(
    "Upload PDF/TXT/DOCX",
    type=["pdf", "txt", "docx"],
    help="Maximum recommended size: 10 pages or 8000 characters"
)

if uploaded is not None:
    with st.spinner("Processing file..."):
        ext = uploaded.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                st.session_state.source_text = read_pdf(uploaded)
            elif ext == "txt":
                st.session_state.source_text = read_txt(uploaded)
            elif ext == "docx":
                st.session_state.source_text = read_docx(uploaded)
            
            st.session_state.source_text = clean_text(st.session_state.source_text)
            st.success(f"Processed {len(st.session_state.source_text)} characters")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Generation Parameters
col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Subject", value="General Knowledge")
with col2:
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"])

num_q = st.slider("Number of Questions", 1, 50, 10)

# Generation Button
if st.button("Generate MCQs", type="primary"):
    if not st.session_state.source_text:
        st.error("Please upload a file before generating.")
    else:
        st.session_state.generate_clicked = True

# MCQ Generation
if st.session_state.generate_clicked:
    with st.spinner("Generating MCQs (this may take a minute)..."):
        try:
            # Initialize components with error handling
            llm = build_llm(GROQ_API_KEY, model_name)
            prepared = summarize_if_needed(llm, st.session_state.source_text)
            chain = build_generation_chain(llm)
            
            # Add slight delay to avoid rate limits
            time.sleep(1)
            
            # Invoke the chain
            result_set = chain.invoke({
                "subject": subject,
                "difficulty": difficulty,
                "n": num_q,
                "content": prepared
            })
            
            # Validate and display results
            if isinstance(result_set, MCQSet):
                # Create dataframe
                questions_data = []
                for i, q in enumerate(result_set.questions):
                    question_dict = {
                        "#": i+1,
                        "question": q.question,
                        "correct": q.correct_label,
                        "explanation": q.explanation
                    }
                    # Add choices
                    for choice in q.choices:
                        question_dict[choice.label] = choice.text
                    questions_data.append(question_dict)
                
                df = pd.DataFrame(questions_data)
                
                # Display results
                st.success(f"Successfully generated {len(df)} MCQs!")
                st.dataframe(df)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result_set.model_dump(), indent=2),
                        file_name="mcqs.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        "Download CSV",
                        data=df.to_csv(index=False),
                        file_name="mcqs.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Unexpected response format from API")
                st.json(result_set)  # Show raw output for debugging
                
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            st.exception(e)  # Show full error details

# Add some helpful tips
st.markdown("""
### Tips for Best Results:
1. Use clear, well-structured source material
2. Keep documents under 10 pages for best performance
3. For large documents, try summarizing first
4. Start with 5-10 questions to test the system
""")