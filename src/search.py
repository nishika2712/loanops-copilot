import os, sys
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv("DB_URL", "postgresql://postgres:dev@localhost:5432/postgres")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = " ".join(sys.argv[1:])
q_emb = model.encode(query)

with psycopg.connect(DB_URL) as conn:
    register_vector(conn)
    rows = conn.execute(
        "SELECT source, content, embedding <=> %s AS distance FROM policy_chunks ORDER BY distance LIMIT 3",
        (q_emb,),
    ).fetchall()

for source, content, dist in rows:
    print(f"\n[{source}] distance={dist:.3f}\n{content}")
