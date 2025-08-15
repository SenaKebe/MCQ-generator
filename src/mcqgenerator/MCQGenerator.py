from pydantic import BaseModel, Field, validator
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class Choice(BaseModel):
    label: str = Field(description="Single-letter label like A, B, C, D")
    text: str

class MCQ(BaseModel):
    question: str
    choices: List[Choice]
    correct_label: str
    explanation: str
    difficulty: str

    @validator("correct_label")
    def label_upper(cls, v):
        v = v.strip().upper()
        if v and v[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return v[0]
        raise ValueError("Correct label must be a single letter")

class MCQSet(BaseModel):
    subject: str
    difficulty: str
    questions: List[MCQ]

def build_llm(api_key: str, model_name: str) -> ChatGroq:
    if not api_key:
        raise ValueError("Missing GROQ API key.")
    return ChatGroq(api_key=api_key, model=model_name, temperature=0.2)

def build_generation_chain(llm: ChatGroq):
    parser = JsonOutputParser(pydantic_object=MCQSet)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert test item writer..."),
        ("human", (
            "Subject: {subject}\n"
            "Difficulty: {difficulty}\n"
            "Number of Questions: {n}\n\n"
            "Source Content: {content}\n\n"
            "Return JSON: {format_instructions}"
        )),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def summarize_if_needed(llm: ChatGroq, text: str) -> str:
    text = text[:8000]  # Hard limit to avoid rate limits
    if len(text) <= 4000:  # Safe for free tier
        return text
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise academic summarizer. Create a concise summary while preserving key facts."),
        ("human", "Summarize the following content:\n\n{content}")
    ])
    
    chain = prompt | llm | (lambda x: x.content)
    return chain.invoke({"content": text})