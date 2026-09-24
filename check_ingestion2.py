from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

all_docs = vectorstore.get(where={"filename": "23501-k20.pdf"})

for meta, content in zip(all_docs["metadatas"], all_docs["documents"]):
    if "Termination of RAN CP interface" in content:
        print(f"--- page {meta.get('page')} (full content, {len(content)} chars) ---")
        print(content)