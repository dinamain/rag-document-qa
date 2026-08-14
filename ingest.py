from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pdfplumber
import os
import re

CHROMA_DIR = "./chroma_db"

# Matches 3GPP's repeating page header as extracted by pdfplumber, e.g.:
# "Release 20 46 3GPP TS 23.501 V20.2.0 (2026-06)"
# (pdfplumber extracts this in reverse order compared to pypdf)
SPEC_HEADER_PATTERN = re.compile(
    r'Release \d+\s+\d+\s+3GPP TS \d+\.\d+ V\d+\.\d+\.\d+ \(\d{4}-\d{2}\)\s*'
)

def clean_text(text: str) -> str:
    text = re.sub(r'(\w)-?\n(\w)', r'\1\2', text)
    text = SPEC_HEADER_PATTERN.sub('', text)
    return text


def table_to_markdown(rows: list) -> str:
    """Convert a pdfplumber-extracted table (list of row lists) to a markdown table."""
    if not rows:
        return ""
    clean_rows = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    header = clean_rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in clean_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_page_content(page) -> str:
    """Extract prose text and tables separately, then combine -- avoids
    tables being garbled inline while also avoiding duplicated content."""
    tables = page.find_tables()
    table_bboxes = [t.bbox for t in tables]

    def not_in_table(obj):
        x0, top, x1, bottom = obj.get("x0"), obj.get("top"), obj.get("x1"), obj.get("bottom")
        if x0 is None:
            return True
        for (tx0, ty0, tx1, ty1) in table_bboxes:
            if x0 >= tx0 - 1 and x1 <= tx1 + 1 and top >= ty0 - 1 and bottom <= ty1 + 1:
                return False
        return True

    if table_bboxes:
        prose_text = page.filter(not_in_table).extract_text() or ""
    else:
        prose_text = page.extract_text() or ""

    table_blocks = []
    for t in tables:
        rows = t.extract()
        md = table_to_markdown(rows)
        if md:
            table_blocks.append(md)

    combined = prose_text
    if table_blocks:
        combined += "\n\n" + "\n\n".join(table_blocks)
    return combined


def load_pdf_as_documents(pdf_path: str) -> list:
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            content = extract_page_content(page)
            documents.append(Document(page_content=content, metadata={"page": i}))
    return documents


def ingest_pdf(pdf_path: str, vectorstore=None):
    filename = os.path.basename(pdf_path)

    if vectorstore is None:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    existing = vectorstore.get(where={"filename": filename})
    if existing and existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
        print(f"Removed {len(existing['ids'])} old chunks for {filename}")

    documents = load_pdf_as_documents(pdf_path)
    print(f"Loaded {len(documents)} pages")

    for doc in documents:
        doc.metadata["filename"] = filename
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        page = chunk.metadata.get("page", "unknown")
        header = f"[Source: {filename}, page {page}]\n"
        chunk.page_content = header + chunk.page_content

    print(f"Created {len(chunks)} chunks")

    vectorstore.add_documents(chunks)
    print(f"Done! {filename} ingested successfully.")