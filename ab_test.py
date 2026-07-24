from query import query_pdf
import query as query_module
from query import rewrite_query as REAL_REWRITE_QUERY

TEST_CASES = [
    {"question": "how do they grade you in this course", "filename": "MANAGEMENT OF SOFTWARE-Ktunotes.in.pdf"},
    {"question": "what did she do with the real time chat thing", "filename": "test.pdf"},
]

def get_chunk_ids(sources):
    return set((s["filename"], s["page"]) for s in sources)

def passthrough(question, llm):
    return question

for case in TEST_CASES:
    print(f"\n{'='*60}\nQUESTION: {case['question']}\n{'='*60}")

    query_module.rewrite_query = REAL_REWRITE_QUERY
    with_rewrite = query_pdf(case["question"], filename=case["filename"])

    query_module.rewrite_query = passthrough
    without_rewrite = query_pdf(case["question"], filename=case["filename"])

    ids_with = get_chunk_ids(with_rewrite["sources"])
    ids_without = get_chunk_ids(without_rewrite["sources"])

    print(f"\nWITH rewrite    -> {with_rewrite['answer'][:150]}")
    print(f"Chunks: {sorted(ids_with)}")
    print(f"\nWITHOUT rewrite -> {without_rewrite['answer'][:150]}")
    print(f"Chunks: {sorted(ids_without)}")

    if ids_with == ids_without:
        print("\n>>> SAME chunk set retrieved (possibly different order)")
    else:
        print(f"\n>>> DIFFERENT — only in WITH: {ids_with - ids_without}, only in WITHOUT: {ids_without - ids_with}")