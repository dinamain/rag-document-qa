"""
Automated eval harness for the RAG pipeline's hallucination-prevention.
Each test case declares whether the system SHOULD answer (content exists)
or SHOULD refuse (content doesn't exist) -- then checks whether the actual
verification STATUS matches that expectation.
"""

from query import query_pdf
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import time
CHROMA_DIR = "./chroma_db"

TEST_CASES = [
    {
        "question": "What is the role of the AMF in the 5G system architecture?",
        "expect": "ANSWER",
        "note": "Core AMF functional description (clause 6.2.1)",
    },
    {
        "question": "What are the standardised SST values and their characteristics?",
        "expect": "ANSWER",
        "note": "Table 5.15.2.2-1",
    },
    {
        "question": "What is a PDU Session and what types exist?",
        "expect": "ANSWER",
        "note": "PDU Session definition + types across specs",
    },
    {
        "question": "What functions does the SMF perform?",
        "expect": "ANSWER",
        "note": "SMF functional description, parallel to AMF (clause 6.2.2 area)",
    },
    {
        "question": "What is an S-NSSAI and what does it consist of?",
        "expect": "ANSWER",
        "note": "Network slice identifier definition (clause 5.15.2)",
    },
    {
        "question": "What information is included in a PDU Session Establishment Request?",
        "expect": "ANSWER",
        "note": "NAS message content, should hit 24501 or 23502 procedures",
    },
    {
        "question": "What is the handover procedure in 4G LTE?",
        "expect": "REFUSE",
        "note": "Tricky: EPS/5GC interworking content exists but doesn't answer this",
    },
    {
        "question": "What is the pricing model for 5G data plans?",
        "expect": "REFUSE",
        "note": "No related content at all",
    },
    {
        "question": "How does Wi-Fi 6 differ from 5G in terms of latency?",
        "expect": "REFUSE",
        "note": "Tricky: real 5G latency content exists but no Wi-Fi 6 comparison",
    },
    {
        "question": "What specific security algorithm is used for NAS ciphering?",
        "expect": "ANSWER",
        "note": "24501 defines the NAS security algorithms IE with named identifiers (e.g. null ciphering algorithm) -- corrected from an earlier mislabeled expectation; full cryptographic algorithm specs still live in TS 33.501, which isn't ingested",
    },
    {
        "question": "What is the maximum transmit power allowed for a 5G base station antenna?",
        "expect": "REFUSE",
        "note": "Hardware/RF specification detail, not covered by these Stage 2/3 procedural specs",
    },
    {
        "question": "What is the exact IP addressing scheme used between the AMF and gNB?",
        "expect": "REFUSE",
        "note": "Tricky: AMF/gNB interfaces are discussed extensively, but exact IP addressing detail is typically deferred to other specs -- tests for confident-sounding but ungrounded specifics",
    },
]



def classify_result(result: dict) -> str:
    """Turn the answer into ANSWER / REFUSE / EMPTY."""
    if result["verification_status"] == "EMPTY_RESPONSE":
        return "EMPTY"
    core = result["answer"].split("⚠️ Note:")[0].strip()
    if not core:
        return "EMPTY"
    if "not covered in the document" in core.lower() and len(core) < 200:
        return "REFUSE"
    return "ANSWER"


def run_eval():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    passed = 0
    results = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(TEST_CASES)}] {case['question']}")
        print(f"Expected: {case['expect']} -- {case['note']}")
        print('='*70)

        result = query_pdf(case["question"], vectorstore=vectorstore)
        actual = classify_result(result)
        answer_matches = (actual == case["expect"])
        verification_ok = not result["hallucinated"]  # verifier should NOT flag hallucination
        status = "PASS" if (answer_matches and verification_ok) else "FAIL"
        if status == "PASS":
            passed += 1

        results.append({
            "question": case["question"],
            "expected": case["expect"],
            "actual": actual,
            "verification_status": result["verification_status"],
            "status": status,
            "answer_preview": result["answer"][:150],
        })

        print(f"\n>>> RESULT: {status} (expected {case['expect']}, got {actual}, verification={result['verification_status']})")
        time.sleep(15)

    print(f"\n\n{'='*70}")
    print(f"SUMMARY: {passed}/{len(TEST_CASES)} passed")
    print('='*70)
    for r in results:
        marker = "✓" if r["status"] == "PASS" else "✗"
        print(f"{marker} [{r['status']}] {r['question'][:60]} (verification: {r['verification_status']})")
        if r["status"] == "FAIL":
            print(f"    expected={r['expected']} actual={r['actual']}")
            print(f"    answer: {r['answer_preview']}")

    return results

if __name__ == "__main__":
    run_eval()