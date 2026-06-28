# RAG Document Q&A System

Upload any PDF and ask questions about it. Get accurate AI-powered answers with source citations — without reading the whole document.

Built with LangChain · ChromaDB · Ollama · FastAPI · React · Docker

---

## What It Does

Most LLMs don't know what's in your private documents. This system solves that using RAG (Retrieval-Augmented Generation):

1. **Upload a PDF** — the document is extracted, chunked, and stored as vector embeddings in ChromaDB
2. **Ask a question** — your question is embedded and compared against stored chunks using semantic similarity search
3. **Get an answer** — the most relevant chunks are retrieved and sent to a local LLM (Llama 3.2 via Ollama) which generates an accurate answer with source citation

The LLM never sees the whole document — only the most relevant sections. This keeps answers focused and grounded.

---

## Architecture

```
PDF Upload
  ↓ PyPDF — extract text
  ↓ RecursiveCharacterTextSplitter — chunk with overlap
  ↓ nomic-embed-text (Ollama) — generate embeddings
  ↓ ChromaDB — store vectors + metadata

User Question
  ↓ nomic-embed-text (Ollama) — embed question
  ↓ ChromaDB — similarity search (top-k chunks)
  ↓ LangChain prompt template — build context prompt
  ↓ Llama 3.2 (Ollama) — generate answer with source citation
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | nomic-embed-text via Ollama |
| Local LLM | Llama 3.2 via Ollama |
| Backend API | FastAPI |
| Frontend | React |
| Containerisation | Docker |

---

## Project Structure

```
rag-document-qa/
├── ingest.py          # PDF ingestion pipeline
├── query.py           # Query and retrieval pipeline
├── main.py            # FastAPI backend (POST /upload, POST /ask)
├── frontend/
│   └── src/
│       └── App.js     # React UI — upload + Q&A
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js
- [Ollama](https://ollama.com) installed and running

### 1. Pull required models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters chromadb fastapi uvicorn pypdf ollama python-multipart
```

### 3. Run the FastAPI backend

```bash
python -m uvicorn main:app --reload
```

API runs at `http://localhost:8000`
Auto-generated docs at `http://localhost:8000/docs`

### 4. Run the React frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload PDF — triggers ingestion pipeline |
| POST | `/ask` | Ask a question — returns answer with source citation |

---

## Key Design Decisions

**Why chunk with overlap?**
Splitting text into fixed chunks risks losing context at boundaries. Overlap ensures sentences that span two chunks remain retrievable in both — no content gets orphaned.

**Why nomic-embed-text instead of the LLM for embeddings?**
Llama 3.2 doesn't support embeddings directly. nomic-embed-text is a dedicated embedding model — faster, more accurate for semantic search.

**Why the same embedding model at index and query time?**
Embeddings are numerical representations learned by a specific model. Switching models between indexing and querying produces incompatible vectors — similarity search returns garbage. Same model both times, always.

**Why RAG over just asking the LLM?**
The LLM was trained on public internet data. Your private PDFs were never part of that training. RAG bridges that gap by retrieving relevant context at query time and giving it to the LLM — no fine-tuning required.

---

## What I Learned Building This

- Discovered a real retrieval failure mode: with `k=3`, answers about multi-section documents were incomplete. Increasing to `k=6` fixed retrieval across longer documents.
- Chunk size and overlap are tunable parameters that directly affect answer quality — not just implementation details.
- FastAPI auto-generates Swagger docs at `/docs` with zero extra work — useful for demonstrating endpoints to stakeholders.

---

## Deployment

*Coming soon — Docker Compose + GitHub Actions CI/CD + Render deployment*

---

## Author

**Dina Usman** — [LinkedIn](https://linkedin.com/in/dina-usman888) · [GitHub](https://github.com/dinamain)
