from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model_name = 'llama-3.3-70b-versatile')

res = llm.invoke("what is capital of india") 

print(res) #it gives everything in the output that includes the content input tokens,output tokens but if you want only the content use the next statement
print(res.content)