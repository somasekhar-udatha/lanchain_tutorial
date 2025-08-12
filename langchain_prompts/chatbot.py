from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

model = ChatHuggingFace(llm = llm,verbose = False)

# res = model.invoke("Who is narendra modi?")
# print(res.content)

# print("Loaded token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


chat_history = [
    SystemMessage(content = "You are a helpful assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    res = model.invoke(chat_history)
    chat_history.append(AIMessage(content = res.content))
    print('AI: ',res.content)

print(chat_history)