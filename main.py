from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import tempfile
from ingest import ingest_pdf
from query import query_pdf
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rag-document-qa-sandy.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    filename: Optional[str] = None

@app.get("/")
def root():
    return {"message": "RAG API is running"}

# Created once at startup, shared by every request
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # basename strips any folder path from the name (blocks "../../" tricks)
    safe_name = os.path.basename(file.filename)
    # a unique temp folder per upload, so simultaneous uploads can't collide
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, safe_name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ingest_pdf(temp_path, vectorstore=vectorstore)
    finally:
        # always clean up, even if ingestion crashes
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"message": f"{safe_name} ingested successfully"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = query_pdf(
        request.question,
        vectorstore=vectorstore,
        filename=request.filename,
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "verification_status": result["verification_status"],
    }