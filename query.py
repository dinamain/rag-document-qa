from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

def query_pdf(question: str):
    # Step 1: Load ChromaDB with same embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
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
    llm = OllamaLLM(model="llama3.2")
    answer = llm.invoke(prompt)

    # Step 6: Show answer with sources
    print("\n--- ANSWER ---")
    print(answer)
    return answer
    

if __name__ == "__main__":
    query_pdf("what projects has this person built?")