from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage("You are a Helpful Assistant"),
    HumanMessage("Tell me about langchain")
]

res = model.invoke(messages)

messages.append(AIMessage(res.content))

print(messages)