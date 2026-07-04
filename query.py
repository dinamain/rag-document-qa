from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
CHROMA_DIR = "./chroma_db"

def query_pdf(question: str, vectorstore=None):
    if vectorstore is None:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    relevant_chunks = retriever.invoke(question)

    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    prompt = f"""You are a precise assistant that answers questions strictly from the provided document context.

Rules:
- Answer ONLY using information explicitly stated in the context below
- If the context contains a partial answer, give that partial answer and state clearly what is missing
- If the context contains no relevant information at all, say exactly: "This topic is not covered in the document."
- Do NOT infer, assume, or use outside knowledge
- Do NOT speculate about what the document might say elsewhere

Context:
{context}

Question: {question}

Answer:"""

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    answer = llm.invoke(prompt)

    sources = [
        {
            "page": chunk.metadata.get("page", "unknown"),
            "filename": chunk.metadata.get("filename", "unknown"),
            "text": chunk.page_content[:150]
        }
        for chunk in relevant_chunks[:3]
    ]

    return {"answer": answer.content, "sources": sources}