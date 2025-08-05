from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1-0528",
    task = "text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

model = ChatHuggingFace(llm = llm,verbose=False)

st.header('Research Tool')

user_input = st.text_input('Enter the prompt')

if st.button('Summarize'):
    res = model.invoke(user_input)
    st.write(res.content)