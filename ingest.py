from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdf(pdf_path: str):
    # Step 1: Load PDF and extract text
    print("Loading PDF...")
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2: Split into chunks
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")


    # Step 3: Create embeddings and store in ChromaDB
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Done! PDF ingested successfully.")

if __name__ == "__main__":
    ingest_pdf("test.pdf")