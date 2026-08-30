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
    -> answer verification (second LLM pass checks the answer against context,
       skipped when the answer is the controlled "not covered" refusal, since
       that is a deterministic non-claim rather than a factual assertion)
    -> final answer + sources returned to user
```

## Key Design Decisions

**Table-aware ingestion (pdfplumber, not plain PyPDF text extraction).** 3GPP specs are dense with tables — parameter definitions, message field tables, cause codes. Plain text extraction flattens tables into unstructured, scrambled text that loses row/column relationships. Tables are now detected separately from prose and converted to clean markdown before chunking, which measurably improved retrieval on table-based questions (verified — see Testing Evidence).

**Header/footer stripping.** Every page carries a repeating boilerplate header (spec number, version, release). Left in place, this diluted the semantic distinctiveness of nearly every chunk, actively hurting both embedding similarity and cross-encoder re-ranking scores. Stripped via a targeted regex before chunking.

**Re-ranking cutoff raised from top_k=6 to top_k=8.** Found via direct testing: the correct AMF functional-description chunk was retrieved and correctly identified as relevant by the cross-encoder, but ranked #7 — just outside the original cutoff. Widening the cutoff fixed this without changing retrieval or ranking logic.

**Hybrid BM25 + vector retrieval — tested, not used.** This was evaluated twice with controlled A/B testing: once on the original portfolio corpus, and again specifically on this 3GPP corpus. Both times, adding BM25 alongside vector search produced identical re-ranking outcomes — the wide k=15 candidate pool plus cross-encoder re-ranking already recovered the correct chunk regardless. Given no measurable benefit and real cost (BM25 rebuilds its index from the full corpus on every query), vector-only retrieval was kept as the evidence-based choice.

**Answer verification layer, and a deliberate exception for controlled refusals.** Prompt instructions alone ("answer only from context, say if not covered") are a soft guardrail — they rely on the model choosing to comply. A second LLM pass explicitly checks the generated answer against the retrieved context and classifies it as FULLY_SUPPORTED, PARTIALLY_SUPPORTED_AND_HONEST, or UNSUPPORTED. Partway through testing, a mid-project LLM provider migration (see below) caused this verifier to start incorrectly flagging every correct refusal as unsupported. The fix: when the answer is the exact controlled sentinel "This topic is not covered in the document.", verification is skipped entirely, since a deterministic refusal is a non-claim, not a factual assertion to check.

**Mid-project model migration (Groq deprecated the original LLM).** The pipeline originally used `llama-3.1-8b-instant`. Groq deprecated this model during development, requiring migration to `openai/gpt-oss-20b`. This surfaced two real, diagnosed issues: (1) the false-refusal-flagging bug described above, fixed by the sentinel-skip logic; and (2) a residual generation issue described honestly in Known Limitations below, which was investigated but not fully resolved.

## Testing Evidence

Built an automated eval harness (`eval_test.py`) covering both directions of correctness across 12 questions:
- 6 questions the specs **should** answer: AMF functional role, standardised SST values from a table, PDU Session definition and types, SMF functional role, S-NSSAI definition, and PDU Session Establishment Request contents
- 6 questions the specs **should not** answer, including several deliberately tricky edge cases where tangentially related content exists but doesn't actually answer the question: 4G LTE handover (only EPS/5GC interworking content exists), 5G data plan pricing (no related content at all), Wi-Fi 6 vs 5G latency (real 5G latency content exists but no Wi-Fi 6 comparison), maximum base station transmit power (hardware/RF detail outside these procedural specs), and exact AMF-gNB IP addressing (interfaces are discussed extensively, but not this level of detail)

**Result: 11/12 passing.** All six "should refuse" cases pass correctly, including every trick case. Five of six "should answer" cases pass with correctly grounded, cited answers. The one failure (NAS ciphering algorithm) is understood and documented in Known Limitations rather than papered over.

## Known Limitations

- **A specific, reproducible overclaim on one question.** For "what specific security algorithm is used for NAS ciphering," the source material states the null ciphering algorithm (5G-EA0) is used *only when configured by the AMF* — a conditional fact. The model consistently drops that condition and states it as an unconditional fact instead. Two fixes were attempted — an explicit prompt instruction to preserve conditional/qualifying language, and increasing the model's `reasoning_effort` from low to medium — neither resolved it. The verification layer correctly and consistently catches this overclaim every time (`UNSUPPORTED`), which is itself working as intended; the underlying generation behavior on this specific question is the unresolved piece.
- BM25 (kept available but disabled) rebuilds its keyword index from the full corpus on every query rather than persisting it — fine at this scale, would need optimization at much larger corpus size.
- Ingestion is non-transactional: old chunks for a file are deleted before new ones are confirmed loaded, so a crash mid-re-ingest could leave a document's chunks missing.
- Answer quality is ultimately bounded by retrieval quality — an obscure or unusually phrased question could still miss the right chunk even with re-ranking.

## What I'd Improve With More Time

- Investigate why `gpt-oss-20b` specifically struggles to preserve conditional language on this question type — likely worth testing few-shot examples in the generation prompt, or a dedicated post-generation pass that checks for and corrects dropped qualifiers.
- Persist the BM25 index instead of rebuilding it per query, in case future corpus growth changes the hybrid-retrieval cost/benefit tradeoff.
- Make ingestion transactional (only replace old chunks after new ones are confirmed written).
- Expand the eval set beyond 12 questions for broader coverage across more clauses and edge cases.
