# RAG Document Q&A System

Upload any PDF and ask questions about it. Get accurate AI-powered answers with source citations — without reading the whole document.

**Live API:** https://rag-document-qa-yrtf.onrender.com  
**API Docs:** https://rag-document-qa-yrtf.onrender.com/docs

Built with LangChain · ChromaDB · FastEmbed · Groq · FastAPI · React · Docker

---

## What It Does

Most LLMs don't know what's in your private documents. This system solves that using RAG (Retrieval-Augmented Generation):

1. **Upload a PDF** — the document is extracted, chunked, and stored as vector embeddings in ChromaDB
2. **Ask a question** — your question is embedded and compared against stored chunks using semantic similarity search
3. **Get an answer** — the most relevant chunks are retrieved and sent to Groq (Llama 3.1) which generates an accurate answer with source citation

The LLM never sees the whole document — only the most relevant sections. This keeps answers focused and grounded.

---

## Architecture

```
PDF Upload
  ↓ PyPDF — extract text
  ↓ RecursiveCharacterTextSplitter — chunk with overlap
  ↓ FastEmbed (BAAI/bge-small-en-v1.5) — generate embeddings
  ↓ ChromaDB — store vectors + metadata

User Question
  ↓ FastEmbed (BAAI/bge-small-en-v1.5) — embed question
  ↓ ChromaDB — similarity search (top-k chunks)
  ↓ LangChain prompt template — build context prompt
  ↓ Groq (Llama 3.1-8b-instant) — generate answer with source citation
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5) |
| LLM | Groq (Llama 3.1-8b-instant) |
| Backend API | FastAPI |
| Frontend | React |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render |

---

## Project Structure

```
rag-document-qa/
├── ingest.py              # PDF ingestion pipeline
├── query.py               # Query and retrieval pipeline
├── main.py                # FastAPI backend (POST /upload, POST /ask)
├── Dockerfile             # Backend container
├── docker-compose.yml     # Multi-container orchestration
├── requirements.txt       # Python dependencies
├── frontend/
│   ├── Dockerfile         # Frontend container
│   └── src/
│       └── App.js         # React UI — upload + Q&A
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI pipeline
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js
- Docker Desktop

### 1. Clone the repo

```bash
git clone https://github.com/dinamain/rag-document-qa.git
cd rag-document-qa
```

### 2. Set up environment variables

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key
```

Get a free key at https://console.groq.com

### 3. Run with Docker Compose

```bash
docker-compose up --build
```

Frontend runs at `http://localhost:3000`  
Backend API runs at `http://localhost:8000`  
API docs at `http://localhost:8000/docs`

### 4. Or run locally without Docker

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
python -m uvicorn main:app --reload
```

Run the React frontend:

```bash
cd frontend
npm install
npm start
```

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

**Why FastEmbed instead of a heavier embedding model?**  
FastEmbed uses ONNX runtime — no PyTorch dependency, under 130MB, runs efficiently on free-tier cloud servers. Heavier models like sentence-transformers require PyTorch (2GB+) which exceeds Render's free tier RAM limit.

**Why the same embedding model at index and query time?**  
Embeddings are numerical representations learned by a specific model. Switching models between indexing and querying produces incompatible vectors — similarity search returns garbage. Same model both times, always.

**Why RAG over just asking the LLM?**  
The LLM was trained on public internet data. Your private PDFs were never part of that training. RAG bridges that gap by retrieving relevant context at query time and giving it to the LLM — no fine-tuning required.

**Why Groq instead of Ollama for deployment?**  
Ollama runs models locally — perfect for development. For cloud deployment, Groq provides fast, free LLM inference via API without needing to run a model on the server.


**Why switch from `langchain_community.vectorstores.Chroma` to `langchain_chroma`?**
The community package is deprecated as of LangChain 0.2.9 and has connection lifecycle bugs on Windows — specifically, it doesn't release SQLite file locks cleanly between requests. Migrating to `langchain-chroma` (the maintained package) resolved persistent `PermissionError: [WinError 32]` errors on upload.

**Why share a single vectorstore instance across requests?**
Creating a new ChromaDB connection per request caused SQLite lock conflicts on Windows when upload and query requests interleaved. Initialising one `Chroma` instance at FastAPI startup and passing it into both `ingest_pdf` and `query_pdf` ensures a single managed connection with no lock contention.

**Why deduplicate by filename on upload?**
Without deduplication, re-uploading the same PDF accumulates duplicate chunks in ChromaDB, polluting similarity search results. On each upload, existing chunks matching that filename are deleted before new ones are added — keeping the vectorstore clean across multiple sessions.

---

## What I Learned Building This

- Discovered a real retrieval failure mode: with `k=3`, answers about multi-section documents were incomplete. Increasing to `k=6` fixed retrieval across longer documents.
- Chunk size and overlap are tunable parameters that directly affect answer quality — not just implementation details.
- FastAPI auto-generates Swagger docs at `/docs` with zero extra work — useful for demonstrating endpoints to stakeholders.
- Ran into an OOM error on Render's free tier caused by PyTorch being installed as a transitive dependency of `sentence-transformers`. Switched to FastEmbed (ONNX-based) to reduce memory footprint from ~2GB to ~130MB.
- Docker networking: containers can't reach each other via `localhost` — used Docker service names and `host.docker.internal` to connect services correctly.
- Debugged a Windows-specific SQLite file lock (`PermissionError: [WinError 32]`) caused by ChromaDB holding connections open between requests — resolved by sharing a single vectorstore instance initialised at startup.
- Migrated from deprecated `langchain_community` Chroma to `langchain_chroma` package after discovering connection lifecycle issues in production.
- Implemented filename-based deduplication to prevent chunk accumulation across multiple upload sessions.
- Discovered MMR retrieval hurts performance on focused academic PDFs by over-diversifying results — plain similarity search with k=5 outperforms it for this use case.
---

## Deployment

**Live API:** https://rag-document-qa-yrtf.onrender.com  
**API Docs:** https://rag-document-qa-yrtf.onrender.com/docs

Deployed on Render. GitHub Actions runs CI on every push to master — installs dependencies and verifies all imports pass before deployment.

---

## Author

**Dina Usman** — [LinkedIn](https://linkedin.com/in/dina-usman888) · [GitHub](https://github.com/dinamain)
