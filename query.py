# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def query_pdf(question: str):
    # Step 1: Load ChromaDB with same embedding model
    # embeddings = HuggingFaceEmbeddings(model="nomic-embed-text", base_url="http://host.docker.internal:11434")
    # llm = Groq(model="llama3.2", base_url="http://host.docker.internal:11434")
    # vectorstore = Chroma(
    #     persist_directory="./chroma_db",
    #     embedding_function=embeddings
    # )
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="all-MiniLM-L6-v2"
    # )
    embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Step 2: Similarity search — find top 3 relevant chunks
    print("Searching for relevant chunks...")
    relevant_chunks = vectorstore.similarity_search(question, k=6)
    print(f"Found {len(relevant_chunks)} relevant chunks")

    # Step 3: Build context from chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # Step 4: Build prompt with context + question
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:"""

    # Step 5: Send to LLM
    print("Sending to LLM...")
    # llm = OllamaLLM(model="llama3.2")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    answer = llm.invoke(prompt)

    # Step 6: Show answer with sources
    print("\n--- ANSWER ---")
    print(answer.content)
    print("\n--- SOURCES ---")
    for chunk in relevant_chunks:
        print(f"Page {chunk.metadata.get('page', 'unknown')}: {chunk.page_content[:100]}...")

    sources = [
    {
        "page": chunk.metadata.get("page", "unknown"),
        "text": chunk.page_content[:150]
    }
    for chunk in relevant_chunks
]

    return {"answer": answer.content, "sources": sources}
    

if __name__ == "__main__":
    query_pdf("what projects has this person built?")