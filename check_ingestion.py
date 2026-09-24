from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

all_docs = vectorstore.get(where={"filename": "23501-k20.pdf"})

# Look for the actual functional-description phrasing 3GPP specs use,
# and the clause number pattern for AMF's dedicated section
search_terms = [
    "AMF supports the following",
    "The AMF includes the following",
    "5.2.1",
]

for term in search_terms:
    matches = [
        (doc_id, content)
        for doc_id, content in zip(all_docs["ids"], all_docs["documents"])
        if term in content
    ]
    print(f"'{term}': {len(matches)} chunk(s)")
    for doc_id, content in matches[:2]:
        print(f"  --- {doc_id} ---")
        print(f"  {content[:300]}")
    print()