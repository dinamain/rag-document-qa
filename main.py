from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from ingest import ingest_pdf
from query import query_pdf

app = FastAPI()
# CORS — allow React frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run ingestion pipeline
    ingest_pdf(temp_path)
    
    # Clean up temp file
    os.remove(temp_path)
    
    return {"message": f"{file.filename} ingested successfully"}

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = query_pdf(request.question)
    return {"answer": answer}