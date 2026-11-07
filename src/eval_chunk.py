import os, json, re
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv("DB_URL", "postgresql://postgres:dev@localhost:5432/postgres")
K = int(os.getenv("K", "3"))
model = SentenceTransformer("all-MiniLM-L6-v2")
norm = lambda s: re.sub(r"\s+", " ", s).lower()
questions = [json.loads(l) for l in open("data/eval/questions.jsonl")]

any_hit, full_hit, rrs, fails = [], [], [], []
with psycopg.connect(DB_URL) as conn:
    register_vector(conn)
    for q in questions:
        if not q["gold_phrases"]:
            continue
        emb = model.encode(q["question"])
        rows = conn.execute(
            "SELECT content FROM policy_chunks ORDER BY embedding <=> %s LIMIT %s", (emb, K)
        ).fetchall()
        chunks = [norm(r[0]) for r in rows]
        phrases = [norm(p) for p in q["gold_phrases"]]
        found = [any(p in c for c in chunks) for p in phrases]
        rank = next((i + 1 for i, c in enumerate(chunks) if any(p in c for p in phrases)), None)
        any_hit.append(any(found)); full_hit.append(all(found))
        rrs.append(1 / rank if rank else 0)
        if not all(found):
            fails.append((q["id"], q["question"], [p for p, f in zip(q["gold_phrases"], found) if not f]))

n = len(any_hit)
print(f"\nChunk-level eval on {n} answerable questions (top {K} chunks)")
print(f"Hit@{K}  (any gold phrase found) : {sum(any_hit)/n:.3f}")
print(f"Full@{K} (all gold phrases found): {sum(full_hit)/n:.3f}")
print(f"MRR (chunk-level)              : {sum(rrs)/n:.3f}")
print(f"\nFailures ({len(fails)}):")
for fid, qq, missing in fails:
    print(f"  Q{fid}: {qq}\n     missing: {missing}")
