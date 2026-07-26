RAG Document Q&A System
# RAG Document Q&A System

Upload any PDF and ask questions about it. Get accurate AI-powered answers with source citations — without reading the whole document.

**Live API:** https://rag-document-qa-yrtf.onrender.com
**API Docs:** https://rag-document-qa-yrtf.onrender.com/docs
**Frontend:** https://rag-document-qa-sandy.vercel.app

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

PDF Upload
↓ PyPDF — extract text
↓ clean_text() — fix words broken across line boundaries
↓ RecursiveCharacterTextSplitter — chunk with overlap
↓ Prepend [Source: filename, page X] header to each chunk
↓ FastEmbed (BAAI/bge-small-en-v1.5) — generate embeddings
↓ ChromaDB — store vectors + metadata

User Question
  ↓ Query rewriting (LLM) — reformulate into a cleaner search query
  ↓ FastEmbed — embed rewritten query
  ↓ Hybrid retrieval: ChromaDB vector search (k=15) + BM25 keyword search (k=15), merged via Reciprocal Rank Fusion
  ↓ Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) — re-score against ORIGINAL question
  ↓ LangChain prompt template — build strict, grounded context prompt
  ↓ Groq (Llama 3.1-8b-instant) — generate answer with source citation
  ↓ Answer verification (LLM) — three-way check: fully supported / honestly partial / unsupported
---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5, ONNX runtime) |
| Re-ranking | sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2) |
| LLM | Groq (Llama 3.1-8b-instant) |
| Backend API | FastAPI |
| Frontend | React |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render (API) · Vercel (frontend) | Keyword Retrieval | BM25 (rank_bm25, via langchain_community) |

---

## Project Structure

rag-document-qa/
├── ingest.py # PDF ingestion: load, clean, chunk, header, embed, store
├── query.py # Query pipeline: rewrite, retrieve, re-rank, generate, verify
├── main.py # FastAPI backend (POST /upload, POST /ask)
├── test_ingest.py # Manual ingestion test script
├── test_query.py # Manual query test script
├── ab_test.py # A/B comparison scripts (e.g. query rewriting on/off)
├── Dockerfile # Backend container
├── docker-compose.yml # Multi-container orchestration
├── requirements.txt # Python dependencies
├── frontend/
│ ├── Dockerfile
│ └── src/App.js # React UI — upload + Q&A
├── .github/workflows/ci.yml # GitHub Actions CI pipeline
├── .gitignore
└── README.md

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
GROQ_API_KEY=your_groq_api_key


Get a free key at https://console.groq.com

### 3. Run with Docker Compose

```bash
docker-compose up --build
```

Frontend: `http://localhost:3000`
Backend API: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### 4. Or run locally without Docker

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

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
Splitting text into fixed chunks risks losing context at boundaries. Overlap ensures sentences that span two chunks remain retrievable in both.

**Why FastEmbed instead of a heavier embedding model?**
FastEmbed uses ONNX runtime — no PyTorch dependency, under 130MB, runs on free-tier cloud servers. Heavier models requiring PyTorch (2GB+) exceed Render's free-tier RAM limit.

**Why the same embedding model at index and query time?**
Embeddings from different models are not comparable — switching models between indexing and querying produces incompatible vectors and garbage similarity scores.

**Why RAG over just asking the LLM?**
The LLM was trained on public data; it has never seen your private PDFs. RAG retrieves relevant context at query time instead of requiring fine-tuning.

**Why Groq instead of a local model for deployment?**
Local models are great for development but need a GPU-backed server to run in production. Groq provides fast, free LLM inference via API without hosting a model.

**Why `langchain_chroma` instead of `langchain_community.vectorstores.Chroma`?**
The community package is deprecated and has connection lifecycle bugs on Windows — it doesn't release SQLite file locks cleanly between requests. Migrating resolved persistent `PermissionError: [WinError 32]` errors on upload.

**Why share a single vectorstore instance across requests?**
Creating a new ChromaDB connection per request caused SQLite lock conflicts when upload and query requests interleaved. One `Chroma` instance, initialized at FastAPI startup, avoids lock contention.

**Why deduplicate by filename on upload?**
Without deduplication, re-uploading the same PDF accumulates duplicate chunks, polluting similarity search. Existing chunks matching a filename are deleted before new ones are added.

**Why scope retrieval by filename?**
Similarity search originally ran across every ingested document — a question about one PDF could silently retrieve chunks from an unrelated one. Metadata filtering on `filename` scopes retrieval correctly; verified in both directions (right document → full answer, unrelated document → correctly "not covered").

**Why rewrite queries before retrieval?**
Casual, vague questions embed differently than formally-worded document text. A/B testing found this makes no measurable difference on keyword-dense documents (like a resume) but produces real retrieval gains on jargon-heavy content, where phrasing divergence from the source document is larger.

**Why re-rank with a cross-encoder after vector search?**
Bi-encoders (used for the initial vector search) embed the query and each chunk separately and compare vectors — fast but comparatively imprecise. A cross-encoder scores the query and a candidate chunk jointly, capturing real interaction between them — more accurate, but too slow to run against an entire vectorstore, so it only re-scores the top candidates from the first pass.

**Why three-way answer verification instead of binary?**
An initial binary supported/not-supported verifier incorrectly flagged an answer that correctly stated what it knew *and* honestly noted what the context didn't cover — punishing intended, honest behavior. Reclassifying into `FULLY_SUPPORTED` / `PARTIALLY_SUPPORTED_AND_HONEST` / `UNSUPPORTED` fixed this.

**Why force `temperature=0`?**
The same question, asked twice, sometimes produced different answers — traced to non-deterministic sampling in the query-rewrite step, which changed which chunks got retrieved. Setting temperature to 0 across all pipeline LLM calls made outputs consistent across repeated identical requests (verified with back-to-back runs).

**Why add hybrid (BM25 + vector) search on top of cross-encoder re-ranking?**
Pure vector similarity is comparatively weak at exact lexical matches — proper nouns, course codes, specific numbers — since embeddings capture semantic meaning, not exact string matches. Hybrid search merges vector search with BM25 keyword matching via Reciprocal Rank Fusion, so a chunk either method considers relevant surfaces near the top even if the other method underweights it.
---

## What I Learned Building This

- **k=3 gave incomplete answers on multi-section PDFs; k=6 fixed it** — retrieval breadth is a real, tunable lever, not just an implementation detail.
- **MMR hurt performance on focused academic PDFs** by over-diversifying results; plain similarity search outperformed it for this use case.
- **An OOM crash on Render's free tier** traced to PyTorch being pulled in as a transitive dependency of a heavier embedding library — switching to FastEmbed's ONNX runtime cut memory from ~2GB to ~130MB.
- **A Windows-specific SQLite file lock** (`PermissionError: [WinError 32]`) was caused by ChromaDB holding connections open between requests — fixed by sharing one vectorstore instance initialized at startup.
- **A PDF text-extraction bug** silently broke words across line boundaries (`"Assessm\nent"` instead of `"Assessment"`) — fixed with a cleanup regex before chunking, discovered while diagnosing weak cross-encoder scores.
- **Re-ranking scores are phrasing-sensitive** — the same chunk scored -9.8 for a casually-phrased question and +3.3 for a more document-aligned phrasing, showing query rewriting and re-ranking are not independent stages.
- **Chunk ordering in the context window affects correctness, not just retrieval quality** — the same three retrieved chunks, in a different order, flipped a correct answer into an incorrect "not covered" response (a real, observed instance of LLM position bias / "lost in the middle").
- **Binary groundedness checks can penalize honesty** — an answer that correctly hedged on missing information was wrongly flagged as unsupported by a binary verifier; three-way classification fixed this.
- **LLM sampling non-determinism affects retrieval, not just wording** — identical questions produced different rewritten queries, different retrieved chunks, and different final answers, until `temperature=0` was applied across the pipeline.
- **Hybrid search's contribution depends on what else is already in the pipeline** — across four controlled tests (a distinctive acronym, an exact numeric mark table twice, and an alphanumeric course code), BM25 measurably improved initial retrieval ranking every time it had something to contribute — in the hardest case, moving the target chunk from position 6 to position 3 in the candidate pool — but never changed the final answer, because a wide k=15 candidate pool plus cross-encoder re-ranking was consistently able to recover the correct chunk regardless. This showed me that a technique can be correctly implemented and genuinely doing its job, while a *different* part of the pipeline (re-ranking, candidate pool width) is absorbing the gap it's meant to close — a more precise finding than either "it helped" or "it didn't."
---

## Deployment

**Live API:** https://rag-document-qa-yrtf.onrender.com
**API Docs:** https://rag-document-qa-yrtf.onrender.com/docs
**Frontend:** https://rag-document-qa-sandy.vercel.app

Deployed on Render (API) and Vercel (frontend). GitHub Actions runs CI on every push to master.

---

## Author

**Dina Usman** — [LinkedIn](https://linkedin.com/in/dina-usman888) · [GitHub](https://github.com/dinamain)