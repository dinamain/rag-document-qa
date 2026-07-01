from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from ingest import ingest_pdf
from query import query_pdf
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
CHROMA_DIR = "./chroma_db"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API is running"}

# Cache at startup
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_path = f"./{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    ingest_pdf(temp_path, vectorstore=vectorstore)  # pass it in
    os.remove(temp_path)
    return {"message": f"{file.filename} ingested successfully"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    result = query_pdf(request.question, vectorstore=vectorstore)
    return {"answer": result["answer"], "sources": result["sources"]}