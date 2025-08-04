from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian Cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many records",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers"
]

query = "tell me about bowler"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

print(cosine_similarity([query_embedding],doc_embeddings))