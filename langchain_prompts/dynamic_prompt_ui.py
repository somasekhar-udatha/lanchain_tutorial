from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate,load_prompt

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

template = load_prompt('template.json')

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button("Summarize"):
    res = model.invoke(prompt)
    st.write(res.content)