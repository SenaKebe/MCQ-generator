import os
from dotenv import load_dotenv
from src.mcqgenerator.logger import logging

# Import Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_core.runnables import RunnableMap

# Load environment variables from the .env file
load_dotenv()

# Access the Gemini API key
key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro"
    google_api_key=key,
    temperature=0.7
)

# ========================= PROMPTS ============================= #

quiz_template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming to the text.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs.
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=quiz_template
)

evaluation_template = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,
you need to evaluate the complexity of the questions and give a complete analysis of the quiz. 
Only use at most 50 words for complexity analysis. 
If the quiz is not at par with the cognitive and analytical abilities of the students,
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.

Quiz_MCQs:
{quiz}
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=evaluation_template
)

# ========================= RUNNABLE CHAINS ============================= #

# Chain to generate quiz
quiz_chain = quiz_generation_prompt | llm

# Chain to review quiz
review_chain = quiz_evaluation_prompt | llm

# RunnableMap that passes outputs properly
generate_evaluate_chain = RunnableMap({
    "quiz": quiz_chain,  # generate the quiz first
    "subject": lambda inputs: inputs["subject"]  # ensure subject is passed to review
}) | review_chain

# ========================= EXECUTION ============================= #

if __name__ == "__main__":
    # Example inputs
    text = "The water cycle consists of evaporation, condensation, precipitation, and collection."
    number = 3
    subject = "Science"
    tone = "simple"
    response_json = """
    {
      "questions": [
        {
          "question": "string",
          "options": ["A", "B", "C", "D"],
          "answer": "string"
        }
      ]
    }
    """

    # Run the chain
    output = generate_evaluate_chain.invoke({
        "text": text,
        "number": number,
        "subject": subject,
        "tone": tone,
        "response_json": response_json
    })

    print("\n=== Generated MCQs ===\n")
    print(output)
