from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import os

def ingest_pdf(pdf_path: str):
    # Step 1: Load PDF and extract text
    print("Loading PDF...")
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    print(f"Loaded {len(documents)} pages")
    # Add filename to metadata of every chunk
    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["filename"] = filename

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
    # embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://host.docker.internal:11434")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="all-MiniLM-L6-v2"
    # )
    embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
    # Load existing vectorstore and ADD to it — not overwrite
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    print(f"Done! {filename} ingested successfully.")


if __name__ == "__main__":
    ingest_pdf("test.pdf")