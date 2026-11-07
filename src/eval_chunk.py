import os, json, re, time
from retrieval import Retriever

K = int(os.getenv("K", "3"))
MODE = os.getenv("MODE", "vector")
FILTER = os.getenv("FILTER", "0")
norm = lambda s: re.sub(r"\s+", " ", s).lower()
questions = [json.loads(l) for l in open(os.getenv("EVAL_FILE", "data/eval/questions.jsonl"))]
r = Retriever()
r.search("warmup", MODE, 1)   # load models before timing

any_hit, full_hit, rrs, fails, lat = [], [], [], [], []
for q in questions:
    if not q["gold_phrases"]:
        continue
    t0 = time.perf_counter()
    res = r.search(q["question"], MODE, K)
    lat.append((time.perf_counter() - t0) * 1000)
    chunks = [norm(x[2]) for x in res]
    phrases = [norm(p) for p in q["gold_phrases"]]
    found = [any(p in c for c in chunks) for p in phrases]
    rank = next((i + 1 for i, c in enumerate(chunks) if any(p in c for p in phrases)), None)
    any_hit.append(any(found)); full_hit.append(all(found))
    rrs.append(1 / rank if rank else 0)
    if not all(found):
        fails.append((q["id"], q["question"], [p for p, f in zip(q["gold_phrases"], found) if not f]))

n = len(any_hit)
lat.sort()
p50, p95 = lat[n // 2], lat[int(0.95 * (n - 1))]
print(f"[{MODE} filter={FILTER}] K={K} n={n}  Hit@{K}={sum(any_hit)/n:.3f}  Full@{K}={sum(full_hit)/n:.3f}  MRR={sum(rrs)/n:.3f}  p50={p50:.0f}ms p95={p95:.0f}ms")
if os.getenv("FAILS"):
    for fid, qq, missing in fails:
        print(f"  Q{fid}: {qq}\n     missing: {missing}")
