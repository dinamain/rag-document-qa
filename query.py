from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

CHROMA_DIR = "./chroma_db"


def rewrite_query(question: str, llm) -> str:
    prompt = f"""Rewrite the following user question into a clear, specific search query 
optimized for finding relevant text in a document via semantic search. 
Keep it concise. Do not answer the question — only rewrite it as a search query.
Return ONLY the rewritten query, nothing else.

User question: {question}

Rewritten query:"""

    result = llm.invoke(prompt)
    return result.content.strip().strip('"')


from fastembed.rerank.cross_encoder import TextCrossEncoder

reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def get_bm25_retriever(vectorstore, filename: str = None, k: int = 15):
    """Build a BM25 keyword retriever over the same chunks stored in ChromaDB.
    Currently unused in query_pdf() -- kept available, see note below."""
    where_filter = {"filename": filename} if filename else None
    raw = vectorstore.get(where=where_filter, include=["documents", "metadatas"])

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def rerank_chunks(question: str, chunks: list, top_k: int = 3) -> list:
    documents = [chunk.page_content for chunk in chunks]
    scores = list(reranker.rerank(question, documents))

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    print("\n--- RERANK SCORES ---")
    for chunk, score in scored_chunks:
        print(f"{score:.4f} | page {chunk.metadata.get('page')}: {chunk.page_content[:60]}")

    return [chunk for chunk, score in scored_chunks[:top_k]]


def verify_answer(question: str, context: str, answer: str, llm) -> dict:
    verification_prompt = f"""You are a strict fact-checker. Classify the ANSWER below relative to the CONTEXT.

Respond in exactly this format:
STATUS: one of FULLY_SUPPORTED, PARTIALLY_SUPPORTED_AND_HONEST, or UNSUPPORTED
REASON: one short sentence explaining why

Definitions:
- FULLY_SUPPORTED: every claim in the answer is directly backed by the context.
- PARTIALLY_SUPPORTED_AND_HONEST: the answer's claims are backed by the context, AND it correctly 
  acknowledges what the context does not cover. This is a GOOD outcome, not a failure.
- UNSUPPORTED: the answer states something as fact that the context does not actually support, 
  without acknowledging the gap. This is a hallucination.

CONTEXT:
{context}

ANSWER:
{answer}
"""
    result = llm.invoke(verification_prompt)
    text = result.content.strip()

    status_line = text.lower().split("reason:")[0]
    is_hallucination = "unsupported" in status_line and "partially" not in status_line

    # Parse the actual status label for downstream use (eval harness, etc.)
    status_label = "UNKNOWN"
    for candidate in ["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED_AND_HONEST", "UNSUPPORTED"]:
        if candidate.lower() in status_line:
            status_label = candidate
            break

    return {"hallucinated": is_hallucination, "raw": text, "status": status_label}

def query_pdf(question: str, vectorstore=None, filename: str = None):
    if vectorstore is None:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    reasoning_effort="medium",   # bumped from low -- low wasn't reliably preserving conditional/qualifier language
    model_kwargs={"include_reasoning": False}
)

    rewritten_question = rewrite_query(question, llm)
    print(f"Original: {question}")
    print(f"Rewritten: {rewritten_question}")

    search_kwargs = {"k": 15}
    if filename:
        search_kwargs["filter"] = {"filename": filename}

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # Hybrid (BM25 + vector) tested again on the 3GPP corpus -- confirmed
    # identical rerank scores to vector-only (page 579 scored 2.3908 either
    # way for the AMF test query), consistent with earlier A/B testing.
    # Reverted to vector-only; get_bm25_retriever() kept available above
    # if this needs revisiting on a different corpus/question set.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever],
        weights=[1.0]
    )

    initial_chunks = hybrid_retriever.invoke(rewritten_question)
    print("\n--- INITIAL RETRIEVAL (before re-rank) ---")
    for c in initial_chunks:
        print(f"page {c.metadata.get('page')}: {c.page_content[:80]}")

    # top_k raised from 6 to 8 -- the correct AMF-definition chunk (page 579)
    # ranked #7 by score, just outside the old top_k=6 cutoff.
    relevant_chunks = rerank_chunks(question, initial_chunks, top_k=8)
    print("\n--- AFTER RE-RANK ---")
    for c in relevant_chunks:
        print(f"page {c.metadata.get('page')}: {c.page_content[:80]}")

    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    prompt = f"""You are a precise assistant that answers questions strictly from the provided document context.

Rules:
- Answer ONLY using information explicitly stated in the context below
- If the context contains a partial answer, give that partial answer and state clearly what is missing
- If the context contains no relevant information at all, say exactly: "This topic is not covered in the document."
- Do NOT infer, assume, or use outside knowledge
- Do NOT speculate about what the document might say elsewhere
- Preserve any conditions, qualifiers, or exceptions stated in the context (e.g. "only when X", "unless Y") -- do not state something as an unconditional fact if the context presents it as conditional or optional

Context:
{context}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt)

    NOT_COVERED_SENTINEL = "This topic is not covered in the document."
    if answer.content.strip() == NOT_COVERED_SENTINEL:
        # A literal refusal is a deterministic non-claim, not a factual
        # assertion -- nothing to verify, so skip the verification LLM
        # call entirely rather than let it (sometimes incorrectly) judge
        # a refusal as if it were an unsupported claim.
        verification = {
            "hallucinated": False,
            "raw": "SKIPPED: exact refusal sentinel matched -- no claims to verify.",
            "status": "FULLY_SUPPORTED",
        }
    else:
        verification = verify_answer(question, context, answer.content, llm)

    print(f"\n--- VERIFICATION ---\n{verification['raw']}")

    final_answer = answer.content
    if verification["hallucinated"]:
        final_answer += "\n\n⚠️ Note: parts of this answer could not be fully verified against the source document."
    sources = [
        {
            "page": chunk.metadata.get("page", "unknown"),
            "filename": chunk.metadata.get("filename", "unknown"),
            "text": chunk.page_content[:150]
        }
        for chunk in relevant_chunks[:3]
    ]

    return {
        "answer": final_answer,
        "sources": sources,
        "verification_status": verification["status"],
        "hallucinated": verification["hallucinated"],
    }