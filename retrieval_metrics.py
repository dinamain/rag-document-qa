from query import rewrite_query, get_bm25_retriever
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers import EnsembleRetriever
from fastembed.rerank.cross_encoder import TextCrossEncoder
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

TEST_CASES = [
    {
        "question": "What is the role of the AMF in the 5G system architecture?",
        "relevant_pages": [("23501-k20.pdf", 579)],
    },
    {
        "question": "What are the standardised SST values and their characteristics?",
        "relevant_pages": [("23501-k20.pdf", 275)],
    },
    {
        "question": "What functions does the SMF perform?",
        "relevant_pages": [("23501-k20.pdf", 224)],
    },
    {
        "question": "What is an S-NSSAI and what does it consist of?",
        "relevant_pages": [("23501-k20.pdf", 273)],
    },
]


def precision_recall_at_k(retrieved_chunks, relevant_pages, k):
    top_k = retrieved_chunks[:k]
    retrieved_set = {(c.metadata.get("filename"), c.metadata.get("page")) for c in top_k}
    relevant_set = set(relevant_pages)

    hits = retrieved_set & relevant_set
    precision = len(hits) / k if k > 0 else 0
    recall = len(hits) / len(relevant_set) if relevant_set else 0
    return precision, recall


def run_metrics():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Note: must use the SAME model as query.py, since ChatGroq is instantiated
    # here directly rather than imported -- keep this in sync with query.py.
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        reasoning_effort="medium",
        model_kwargs={"include_reasoning": False}
    )

    print(f"{'Question':<55} {'P@15':>7} {'R@15':>7} {'P@8':>7} {'R@8':>7}")
    print("-" * 85)

    for case in TEST_CASES:
        rewritten = rewrite_query(case["question"], llm)

        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        hybrid_retriever = EnsembleRetriever(retrievers=[vector_retriever], weights=[1.0])
        initial_chunks = hybrid_retriever.invoke(rewritten)

        documents = [c.page_content for c in initial_chunks]
        scores = list(reranker.rerank(case["question"], documents))
        reranked = [c for c, s in sorted(zip(initial_chunks, scores), key=lambda x: x[1], reverse=True)]

        p15, r15 = precision_recall_at_k(initial_chunks, case["relevant_pages"], 15)
        p8, r8 = precision_recall_at_k(reranked, case["relevant_pages"], 8)

        print(f"{case['question'][:53]:<55} {p15:>7.3f} {r15:>7.3f} {p8:>7.3f} {r8:>7.3f}")


if __name__ == "__main__":
    run_metrics()