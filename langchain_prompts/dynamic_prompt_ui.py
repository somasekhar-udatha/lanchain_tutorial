from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1-0528",
    task = "text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

model = ChatHuggingFace(llm = llm,verbose = False)

st.header('Research Tool')

paper_input = st.text_input('Enter the name of the paper')
style_input = st.selectbox("Select Explanation style",['Beginner Friendly','Technical','Core oriented',"Mathematical"])
length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (detailed explanation)"])

template = PromptTemplate(
    template = """Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
    - Include relevant mathematical equations if present in the paper.  
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
    - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input','style_input','length_input']
)

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button("Summarize"):
    res = model.invoke(prompt)
    st.write(res.content)