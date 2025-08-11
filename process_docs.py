# process_docs.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

def load_pdfs(directory="data"):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(directory, filename)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                texts.append((filename, text))
    return texts

def preprocess_and_index():
    # Load documents
    raw_docs = load_pdfs()

    # Split and embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_chunks = []
    metadata = []

    for doc_name, text in raw_docs:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": doc_name}] * len(chunks))

    embeddings = model.encode(all_chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Save index + data
    faiss.write_index(index, "faiss_index/index.faiss")
    np.save("faiss_index/chunks.npy", np.array(all_chunks, dtype=object))
    np.save("faiss_index/metadata.npy", np.array(metadata, dtype=object))

    print("✅ Index built and saved.")

if __name__ == "__main__":
    preprocess_and_index()
