import os, json, sys
from collections import defaultdict
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv("DB_URL", "postgresql://postgres:dev@localhost:5432/postgres")
K = 5
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [json.loads(l) for l in open("data/eval/questions.jsonl")]

def retrieve(conn, q, k=K):
    emb = model.encode(q)
    return conn.execute(
        "SELECT source, embedding <=> %s AS d FROM policy_chunks ORDER BY d LIMIT %s",
        (emb, k),
    ).fetchall()

hits3, rrs, recalls = [], [], []
by_type = defaultdict(list)
failures, unans_dists, ans_dists = [], [], []

with psycopg.connect(DB_URL) as conn:
    register_vector(conn)
    for q in questions:
        rows = retrieve(conn, q["question"])
        sources = [r[0] for r in rows]
        top_dist = rows[0][1]
        gold = set(q["gold_sources"])

        if not gold:                      # unanswerable question
            unans_dists.append(top_dist)
            continue

        ans_dists.append(top_dist)
        rank = next((i + 1 for i, s in enumerate(sources) if s in gold), None)
        hit = rank is not None and rank <= 3
        rr = 1 / rank if rank else 0
        recall = len(gold & set(sources[:3])) / len(gold)

        hits3.append(hit); rrs.append(rr); recalls.append(recall)
        by_type[q["type"]].append(hit)
        if not hit or recall < 1:
            failures.append((q["id"], q["question"], sources[:3], list(gold)))

n = len(hits3)
print(f"\nAnswerable questions: {n}")
print(f"Hit@3    : {sum(hits3)/n:.3f}")
print(f"MRR      : {sum(rrs)/n:.3f}")
print(f"Recall@3 : {sum(recalls)/n:.3f}")

print("\nHit@3 by type:")
for t, v in by_type.items():
    print(f"  {t:<10} {sum(v)}/{len(v)}")

print("\nTop-1 distance (lower = closer):")
print(f"  answerable   avg {sum(ans_dists)/len(ans_dists):.3f}  max {max(ans_dists):.3f}")
print(f"  unanswerable avg {sum(unans_dists)/len(unans_dists):.3f}  min {min(unans_dists):.3f}")

print(f"\nFailures / partial ({len(failures)}):")
for fid, qq, got, gold in failures:
    print(f"  Q{fid}: {qq}\n     got={got} gold={gold}")
