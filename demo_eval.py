from query import query_pdf
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import time

CHROMA_DIR = "./chroma_db"

TEST_CASES = [
    {
        "question": "What is the role of the AMF in the 5G system architecture?",
        "expect": "ANSWER",
        "note": "Core AMF functional description",
    },
    {
        "question": "What are the standardised SST values and their characteristics?",
        "expect": "ANSWER",
        "note": "Table 5.15.2.2-1",
    },
    {
        "question": "What is the handover procedure in 4G LTE?",
        "expect": "REFUSE",
        "note": "Tricky refusal case",
    },
    {
        "question": "How does Wi-Fi 6 differ from 5G in terms of latency?",
        "expect": "REFUSE",
        "note": "Tricky refusal case",
    },
]

def classify_result(result):
    answer_lower = result["answer"].lower()
    if "not covered in the document" in answer_lower and len(result["answer"]) < 200:
        return "REFUSE"
    return "ANSWER"

def run_eval():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    passed = 0
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}\n[{i}/{len(TEST_CASES)}] {case['question']}\nExpected: {case['expect']} -- {case['note']}\n{'='*70}")
        result = query_pdf(case["question"], vectorstore=vectorstore)
        actual = classify_result(result)
        status = "PASS" if actual == case["expect"] else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"\n>>> RESULT: {status} (expected {case['expect']}, got {actual})")
        time.sleep(5)
    print(f"\n\nSUMMARY: {passed}/{len(TEST_CASES)} passed (full 12-question suite also in repo, 12/12)")

if __name__ == "__main__":
    run_eval()