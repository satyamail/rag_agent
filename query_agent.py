# query_agent.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Load index and data
index = faiss.read_index("faiss_index/index.faiss")
chunks = np.load("faiss_index/chunks.npy", allow_pickle=True)
metadata = np.load("faiss_index/metadata.npy", allow_pickle=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

def search_context(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]

def generate_answer(query):
    context_chunks = search_context(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    # Call Ollama
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: LLM request failed."
