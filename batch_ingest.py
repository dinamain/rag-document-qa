"""
Batch-ingests every PDF in a folder using the existing ingest_pdf() function.
Reuses one shared vectorstore connection across all files for speed.
"""

import os
from pathlib import Path
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from ingest import ingest_pdf, CHROMA_DIR

PDF_DIR = Path("./specs_pdf")  # folder with your converted PDFs

def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR.resolve()}")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting batch ingestion...\n")

    # Create one shared vectorstore connection, reused for every file
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            ingest_pdf(str(pdf_path), vectorstore=vectorstore)
        except Exception as e:
            print(f"  FAILED: {pdf_path.name} -> {e}")
        print()

    print("Batch ingestion complete.")


if __name__ == "__main__":
    main()