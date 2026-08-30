# RAG Document Q&A System

Upload any PDF and ask questions about it. Get accurate AI-powered answers with source citations — without reading the whole document.

**Repo:** https://github.com/dinamain/rag-document-qa
**Status:** Feature-complete — full pipeline built and tested. No live demo (see Known Limitations).

Built with LangChain · ChromaDB · FastEmbed · Groq · FastAPI · React · Docker

---

## What It Does

Most LLMs don't know what's in your private documents. This system solves that using RAG (Retrieval-Augmented Generation):

1. **Upload a PDF** — the document is extracted (prose and tables handled separately), chunked, and stored as vector embeddings in ChromaDB
2. **Ask a question** — your question is embedded and compared against stored chunks using semantic similarity search
3. **Get an answer** — the most relevant chunks are retrieved and sent to Groq (`openai/gpt-oss-20b`) which generates an accurate answer with source citation

The LLM never sees the whole document — only the most relevant sections. This keeps answers focused and grounded.

---

## Architecture

```
PDF Upload
  ↓ pdfplumber — extract text; tables detected and converted to markdown
      separately from prose, so column/row structure survives extraction
  ↓ clean_text() — fix words broken across line boundaries; strip repeating
      page headers/footers that otherwise dilute chunk relevance
  ↓ RecursiveCharacterTextSplitter — chunk with overlap
  ↓ Prepend [Source: filename, page X] header to each chunk
  ↓ FastEmbed (BAAI/bge-small-en-v1.5) — generate embeddings
  ↓ ChromaDB — store vectors + metadata

User Question
  ↓ Query rewriting (LLM) — reformulate into a cleaner search query
  ↓ FastEmbed — embed rewritten query
  ↓ Vector retrieval (ChromaDB similarity search, k=15)
      Note: hybrid retrieval (BM25 + vector via Reciprocal Rank Fusion) was
      implemented and A/B tested (see Key Design Decisions) but was removed
      from the active pipeline to reduce memory footprint — vector-only
      retrieval is the current path
  ↓ Cross-encoder re-ranking (FastEmbed ONNX, Xenova/ms-marco-MiniLM-L-6-v2) — re-score against ORIGINAL question, top 8
  ↓ LangChain prompt template — build strict, grounded context prompt
  ↓ Groq (openai/gpt-oss-20b) — generate answer with source citation
  ↓ Answer verification (LLM) — three-way check: fully supported / honestly
      partial / unsupported. Skipped when the answer is the exact controlled
      refusal sentinel ("This topic is not covered in the document."), since
      that's a deterministic non-claim rather than a factual assertion —
      see Key Design Decisions for why this exception exists.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| PDF Extraction | pdfplumber (table-aware) |
| Vector Database | ChromaDB |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5, ONNX runtime) |
| Re-ranking | FastEmbed (Xenova/ms-marco-MiniLM-L-6-v2, ONNX runtime) |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Backend API | FastAPI |
| Frontend | React |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
rag-document-qa/
├── ingest.py                  # PDF ingestion: extract (prose + tables), clean, chunk, header, embed, store
├── query.py                   # Query pipeline: rewrite, retrieve, re-rank, generate, verify
├── main.py                    # FastAPI backend (POST /upload, POST /ask)
├── eval_test.py                # Automated eval harness (12 questions, checks answer + verification status)
├── test_ingest.py             # Manual ingestion test script
├── test_query.py              # Manual query test script
├── ab_test.py                 # A/B comparison scripts (e.g. query rewriting on/off)
├── Dockerfile                 # Backend container
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
├── frontend/
│   ├── Dockerfile
│   └── src/App.js             # React UI — upload + Q&A
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
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

**Why pdfplumber instead of plain PyPDF text extraction?**
Documents with dense tables (parameter tables, field definitions, structured reference data) get scrambled by plain text extraction — cell contents run together and lose their row/column relationships. pdfplumber detects table regions separately from prose, and tables are converted to clean markdown before chunking, preserving structure that plain extraction destroys.

**Why strip repeating page headers/footers before chunking?**
Documents with a repeating boilerplate header or footer on every page (title, version, page number) end up embedding that near-identical text into nearly every chunk. This dilutes each chunk's semantic distinctiveness, measurably hurting both embedding similarity and cross-encoder re-ranking — verified by comparing retrieval quality before and after stripping the pattern.

**Why re-rank with a cutoff of top 8, not top 6?**
Testing surfaced a case where the single most relevant chunk for a question was correctly identified as relevant by the cross-encoder, but ranked #7 — just outside a top_k=6 cutoff — causing the final answer to miss it. Widening the cutoff to 8 fixed this without changing retrieval or ranking logic.

**Why FastEmbed instead of a heavier embedding model?**
FastEmbed uses ONNX runtime — no PyTorch dependency, under 130MB, runs on free-tier cloud servers. Heavier models requiring PyTorch (2GB+) exceed Render's free-tier RAM limit.

**Why the same embedding model at index and query time?**
Embeddings from different models are not comparable — switching models between indexing and querying produces incompatible vectors and garbage similarity scores.

**Why RAG over just asking the LLM?**
The LLM was trained on public data; it has never seen your private PDFs. RAG retrieves relevant context at query time instead of requiring fine-tuning.

**Why Groq instead of a local model for deployment?**
Local models are great for development but need a GPU-backed server to run in production. Groq provides fast, free LLM inference via API without hosting a model.

**Why migrate from `llama-3.1-8b-instant` to `openai/gpt-oss-20b` mid-project?**
Groq deprecated `llama-3.1-8b-instant` during development, forcing a mid-project model swap. This surfaced a real regression: the verification step started incorrectly flagging every correct "not covered" refusal as an unsupported hallucination, because the new model's judgment on refusals was miscalibrated — it tended to treat any loosely-related content in the context as proof the topic was "covered," even when that content didn't actually answer the question. Fixed by skipping verification entirely for the exact controlled refusal sentinel (see below), since a deterministic refusal isn't a factual claim to check in the first place.

**Why skip verification for the exact "not covered" sentinel string?**
A literal, controlled refusal ("This topic is not covered in the document.") makes no factual assertion — there's nothing in it to check against the context. Running it through the verifier anyway (as the pipeline originally did) meant the verifier was effectively being asked "is the absence of an answer supported by the context," a category error that a stricter model answered inconsistently and often incorrectly. Detecting the exact sentinel and skipping verification for it removes that entire failure mode.

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

**Why switch the re-ranker from `sentence-transformers` to FastEmbed's ONNX cross-encoder?**
The original re-ranker used `sentence-transformers`, which pulls in a full PyTorch install alongside FastEmbed's ONNX embeddings — running two heavy runtimes together caused an out-of-memory crash on Render's free tier. Switching to FastEmbed's own ONNX-based cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, the same underlying model, ONNX-converted) fixed the memory issue. Verified the swap didn't change behavior by re-running the same test questions and confirming identical re-rank scores to four decimal places — the ONNX conversion introduced no measurable numerical drift.

**Why three-way answer verification instead of binary?**
An initial binary supported/not-supported verifier incorrectly flagged an answer that correctly stated what it knew *and* honestly noted what the context didn't cover — punishing intended, honest behavior. Reclassifying into `FULLY_SUPPORTED` / `PARTIALLY_SUPPORTED_AND_HONEST` / `UNSUPPORTED` fixed this.

**Why force `temperature=0`?**
The same question, asked twice, sometimes produced different answers — traced to non-deterministic sampling in the query-rewrite step, which changed which chunks got retrieved. Setting temperature to 0 across all pipeline LLM calls made outputs consistent across repeated identical requests (verified with back-to-back runs).

**Why build hybrid (BM25 + vector) search, then remove it from the active pipeline?**
Pure vector similarity is comparatively weak at exact lexical matches — proper nouns, course codes, specific numbers — since embeddings capture semantic meaning, not exact string matches. Hybrid search (vector + BM25 keyword matching merged via Reciprocal Rank Fusion) was built and A/B tested to close that gap. Across multiple controlled tests — including a fresh, separate test on a different, denser technical corpus — it never changed the *final answer*, because a wide k=15 candidate pool plus cross-encoder re-ranking was already recovering the correct chunk regardless. Meanwhile BM25 rebuilding a full in-memory index on every query pushed peak memory over Render's free-tier 512MB limit. Given no measurable answer-quality gain and a real memory cost, hybrid retrieval was removed from the active query path; vector-only retrieval + re-ranking is what runs today. The implementation remains recoverable from git history for a higher-memory deployment.

---

## What I Learned Building This

- **k=3 gave incomplete answers on multi-section PDFs; k=6 fixed it** — retrieval breadth is a real, tunable lever, not just an implementation detail.
- **MMR hurt performance on focused academic PDFs** by over-diversifying results; plain similarity search outperformed it for this use case.
- **An OOM crash on Render's free tier** traced to PyTorch being pulled in as a transitive dependency of a heavier embedding library — switching to FastEmbed's ONNX runtime cut memory from ~2GB to ~130MB. A second, later OOM — from adding a PyTorch-based cross-encoder reranker alongside FastEmbed's ONNX embeddings — was fixed the same way, by switching the reranker to FastEmbed's own ONNX cross-encoder.
- **A Windows-specific SQLite file lock** (`PermissionError: [WinError 32]`) was caused by ChromaDB holding connections open between requests — fixed by sharing one vectorstore instance initialized at startup.
- **A PDF text-extraction bug** silently broke words across line boundaries (`"Assessm\nent"` instead of `"Assessment"`) — fixed with a cleanup regex before chunking, discovered while diagnosing weak cross-encoder scores.
- **Repeating page headers/footers quietly pollute every chunk** on documents with a fixed page template — fixed by stripping the repeating pattern before chunking, discovered while diagnosing why a chunk containing the correct answer was consistently ranked lower than it should have been.
- **A correct, retrieved chunk can still miss the final answer if the re-rank cutoff is too tight** — found by tracing a specific failed answer back to its source chunk, confirming it was retrieved and correctly scored as relevant, just one rank outside the cutoff.
- **Plain text extraction scrambles tabular data** — column/row relationships are lost when a table is flattened into a single text stream; switching to structure-aware extraction (tables handled separately from prose, converted to markdown) fixed retrieval and answer accuracy on table-based questions.
- **Re-ranking scores are phrasing-sensitive** — the same chunk scored -9.8 for a casually-phrased question and +3.3 for a more document-aligned phrasing, showing query rewriting and re-ranking are not independent stages.
- **Chunk ordering in the context window affects correctness, not just retrieval quality** — the same three retrieved chunks, in a different order, flipped a correct answer into an incorrect "not covered" response (a real, observed instance of LLM position bias / "lost in the middle").
- **Binary groundedness checks can penalize honesty** — an answer that correctly hedged on missing information was wrongly flagged as unsupported by a binary verifier; three-way classification fixed this.
- **LLM sampling non-determinism affects retrieval, not just wording** — identical questions produced different rewritten queries, different retrieved chunks, and different final answers, until `temperature=0` was applied across the pipeline.
- **A technique can work correctly and still be the wrong choice for the deployment target** — hybrid search measurably improved retrieval ranking in isolation, but a different part of the pipeline (wide candidate pool + re-ranking) was already absorbing the gap it closed, and it cost more memory than the free tier allowed. Building and testing a feature is a separate question from whether it earns its keep in production.
- **Every added pipeline stage has a resource cost somewhere, not just a speed one** — hybrid search, re-ranking, and running two ONNX models simultaneously each add real memory or latency overhead; a technique being correct doesn't mean it's free to deploy.
- **A provider deprecating a model mid-project can silently change more than availability** — swapping `llama-3.1-8b-instant` for `openai/gpt-oss-20b` after Groq's deprecation didn't just require a model-string change; the new model's judgment calibration on the verification task was measurably different, and it took a full eval re-run to catch.
- **Not every generation flaw is fixable by prompting alone** — a specific model behavior (dropping conditional/qualifying language when stating a fact) persisted even after an explicit prompt instruction and a higher reasoning-effort setting. Documenting a known, understood limitation honestly is preferable to claiming a fix that doesn't actually work.

---

## Known Limitations

**No live demo — deployed via GitHub only.** This project was deployed on Render's free tier during development, but the full pipeline (cross-encoder re-ranking plus embeddings, two ONNX models loaded simultaneously) pushes close to the free tier's 512MB memory limit, and hybrid BM25 search specifically would exceed it (see above — this is why it was removed from the active pipeline). Rather than strip working, well-tested code just to fit a memory-constrained free tier, this project is presented as source code with full local setup instructions above — see "Getting Started." A production deployment would use a higher-memory tier (Render Starter, $7/month, or equivalent) to run the full pipeline, including hybrid search, reliably.

**BM25 index (when reintroduced) would be rebuilt on every query, not persisted.** The hybrid search implementation (recoverable from git history) called `vectorstore.get(...)` to pull all matching chunks from ChromaDB and rebuild an in-memory `BM25Retriever` from scratch on every single query. This is fine at portfolio scale (tens of chunks per document) but would not hold up at real scale — at 100k+ documents, re-fetching and re-tokenizing the entire corpus per query becomes a real bottleneck. The fix: cache the BM25 index in memory at startup and only rebuild it on ingestion, not on every query. At genuine production scale, this would be replaced with a disk-persisted, incrementally-updatable keyword search engine (Elasticsearch, OpenSearch) rather than an in-memory rebuild.

**Non-transactional ingestion.** Re-ingesting a document deletes its existing chunks before confirming the new ones have loaded successfully — an ingestion crash mid-re-ingest can leave a document's chunks permanently missing. A production version would stage new chunks, verify them, then delete the old ones only after the new ones are confirmed.

**A specific, reproducible overclaim on one question, tied to the current LLM.** When asked what security algorithm is used for NAS ciphering, `openai/gpt-oss-20b` consistently states the null ciphering algorithm (5G-EA0) as an unconditional fact, even though the source material presents it as conditional ("used only when configured by the AMF"). An explicit prompt instruction to preserve conditional language, and increasing `reasoning_effort` from low to medium, both failed to fix this. The verification layer correctly and consistently flags this specific answer as unsupported, which is the system working as designed — the residual issue is in generation, not detection.

---

## Deployment

This project was deployed and tested on Render (API) and Vercel (frontend) during development — the code and configuration are proven to work end-to-end. See **Known Limitations** above for why no permanent live demo link is kept up.

GitHub Actions runs CI on every push to master.

---

## Author

**Dina Usman** — [LinkedIn](https://linkedin.com/in/dina-usman888) · [GitHub](https://github.com/dinamain)
