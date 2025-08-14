import pandas as pd
import json
import streamlit as st
from src.mcqgenerator.utils import read_file
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

# Streamlit app title
st.title("MCQs Creator Application with Gemini 🦜🔗")

# Form for user input
with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or TXT file")

    # Additional form fields
    mcq_count = st.number_input("Number of MCQs", min_value=1, max_value=50)
    subject = st.text_input("Subject")
    tone = st.selectbox("Quiz Tone", ["Simple", "Moderate", "Hard"])
    button = st.form_submit_button("Create MCQs")

# Generate MCQs on button click
if button:
    if uploaded_file is None:
        st.warning("Please upload a file first!")
    elif not subject:
        st.warning("Please enter a subject!")
    else:
        with st.spinner("Generating MCQs..."):
            try:
                # Read text from UploadedFile
                if uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    st.error("Unsupported file type!")
                    text = None

                if not text:
                    st.error("No text found in the uploaded file!")
                else:
                    # Run Gemini MCQ generator
                    response = generate_evaluate_chain.invoke(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone.lower(),
                            "response_json": json.dumps({
                                "questions": [
                                    {
                                        "question": "string",
                                        "options": ["A", "B", "C", "D"],
                                        "answer": "string"
                                    }
                                ]
                            })
                        }
                    )

                    # Extract JSON from response
                    content_str = response.content if hasattr(response, "content") else str(response)
                    start = content_str.find("{")
                    end = content_str.rfind("}") + 1
                    quiz_json_str = content_str[start:end]
                    quiz_data = json.loads(quiz_json_str)

                    # Convert to table
                    table_data = []
                    for q in quiz_data.get("questions", []):
                        table_data.append({
                            "Question": q.get("question"),
                            "Option A": q.get("options")[0] if len(q.get("options", [])) > 0 else "",
                            "Option B": q.get("options")[1] if len(q.get("options", [])) > 1 else "",
                            "Option C": q.get("options")[2] if len(q.get("options", [])) > 2 else "",
                            "Option D": q.get("options")[3] if len(q.get("options", [])) > 3 else "",
                            "Answer": q.get("answer")
                        })

                    # Display table
                    if table_data:
                        df = pd.DataFrame(table_data)
                        df.index += 1
                        st.subheader("Generated MCQs")
                        st.table(df)
                    else:
                        st.warning("No quiz data found in the model response.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
