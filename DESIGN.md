# RAG Chatbot for 3GPP Telecom Standards — Design Doc

## Problem & Goal

This system is a Retrieval-Augmented Generation (RAG) chatbot built over 3GPP 5G specifications — TS 23.501 (System Architecture), TS 23.502 (Procedures), and TS 24.501 (NAS Protocol). The core goal is **minimal to near-zero hallucination**: the system should answer confidently and correctly when a spec covers the question, and refuse honestly when it doesn't — rather than confidently making something up either way.

## Architecture Overview

```
PDF ingestion (pdfplumber, table-aware)
    -> header/footer cleanup
    -> chunking (1000 chars, 150 overlap)
    -> FastEmbed embeddings (BAAI/bge-small-en-v1.5)
    -> ChromaDB vector store
    -> [query time]
    -> query rewriting (LLM rephrases user question into a search query)
    -> vector similarity retrieval (k=15)
    -> cross-encoder re-ranking (top 8)
    -> LLM answer generation, grounded strictly in retrieved context
    -> answer verification (second LLM pass checks the answer against context)
    -> final answer + sources returned to user
```

## Key Design Decisions

**Table-aware ingestion (pdfplumber, not plain PyPDF text extraction).** 3GPP specs are dense with tables — parameter definitions, message field tables, cause codes. Plain text extraction flattens tables into unstructured, scrambled text that loses row/column relationships. Tables are now detected separately from prose and converted to clean markdown before chunking, which measurably improved retrieval on table-based questions (verified — see Testing Evidence).

**Header/footer stripping.** Every page carries a repeating boilerplate header (spec number, version, release). Left in place, this diluted the semantic distinctiveness of nearly every chunk, actively hurting both embedding similarity and cross-encoder re-ranking scores. Stripped via a targeted regex before chunking.

**Re-ranking cutoff raised from top_k=6 to top_k=8.** Found via direct testing: the correct AMF functional-description chunk was retrieved and correctly identified as relevant by the cross-encoder, but ranked #7 — just outside the original cutoff. Widening the cutoff fixed this without changing retrieval or ranking logic.

**Hybrid BM25 + vector retrieval — tested, not used.** This was evaluated twice with controlled A/B testing: once on the original portfolio corpus, and again specifically on this 3GPP corpus. Both times, adding BM25 alongside vector search produced identical re-ranking outcomes — the wide k=15 candidate pool plus cross-encoder re-ranking already recovered the correct chunk regardless. Given no measurable benefit and real cost (BM25 rebuilds its index from the full corpus on every query), vector-only retrieval was kept as the evidence-based choice.

**Answer verification layer.** Prompt instructions alone ("answer only from context, say if not covered") are a soft guardrail — they rely on the model choosing to comply. A second LLM pass explicitly checks the generated answer against the retrieved context and classifies it as FULLY_SUPPORTED, PARTIALLY_SUPPORTED_AND_HONEST, or UNSUPPORTED. This is the actual mechanism behind "near-zero hallucination," not just a prompt hoping for good behavior.

## Testing Evidence

Built an automated eval harness (`eval_test.py`) covering both directions of correctness:
- 3 questions the specs **should** answer (AMF functional role, standardised SST values from a table, PDU Session types)
- 3 questions the specs **should not** answer, including two deliberately tricky edge cases where tangentially related content exists but doesn't actually answer the question (4G LTE handover — only EPS/5GC interworking content exists; Wi-Fi 6 vs 5G latency — real 5G latency content exists but no Wi-Fi 6 comparison)

**Result: 6/6 passing.** The system answered correctly and cited sources on all three "should answer" cases, and correctly refused without fabricating information on all three "should refuse" cases, including both trick cases.

## Known Limitations

- BM25 (kept available but disabled) rebuilds its keyword index from the full corpus on every query rather than persisting it — fine at this scale, would need optimization at much larger corpus size.
- Ingestion is non-transactional: old chunks for a file are deleted before new ones are confirmed loaded, so a crash mid-re-ingest could leave a document's chunks missing.
- Answer quality is ultimately bounded by retrieval quality — an obscure or unusually phrased question could still miss the right chunk even with re-ranking.

## What I'd Improve With More Time

- Persist the BM25 index instead of rebuilding it per query, in case future corpus growth changes the hybrid-retrieval cost/benefit tradeoff.
- Make ingestion transactional (only replace old chunks after new ones are confirmed written).
- Expand the eval set beyond 12 questions for broader coverage across more clauses and edge cases.
