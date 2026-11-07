import os, json, re
from retrieval import Retriever

norm = lambda s: re.sub(r"\s+", " ", s).lower()
qs = [json.loads(l) for l in open("data/eval/questions.jsonl")]
r = Retriever()

def hit(q, mode, f, k=1):
    os.environ["FILTER"] = f
    ch = [norm(x[2]) for x in r.search(q["question"], mode, k)]
    return all(any(norm(p) in c for c in ch) for p in q["gold_phrases"])

for mode in ("hybrid", "rerank"):
    print(f"\n{mode}: flips when filter is turned on (K=1)")
    for q in qs:
        if not q["gold_phrases"]:
            continue
        a, b = hit(q, mode, "0"), hit(q, mode, "1")
        if a != b:
            print(f"  Q{q['id']} {'GAINED' if b else 'LOST'}: {q['question']}")

os.environ["FILTER"] = "1"
for qid in (11, 19):
    q = next(x for x in qs if x["id"] == qid)
    print(f"\nQ{qid}: {q['question']}\n  gold: {q['gold_phrases']}")
    for i, x in enumerate(r.search(q["question"], "rerank", 3), 1):
        print(f"  {i}. {x[1]} | {' '.join(x[2].split())[:110]}")
