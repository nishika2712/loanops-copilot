import os, glob
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv("DB_URL", "postgresql://postgres:dev@localhost:5432/postgres")
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, size=400, overlap=80):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]

with psycopg.connect(DB_URL, autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS policy_chunks")
    conn.execute("""
        CREATE TABLE policy_chunks (
            id SERIAL PRIMARY KEY,
            source TEXT,
            chunk_index INT,
            content TEXT,
            embedding vector(384)
        )""")

    total = 0
    for path in sorted(glob.glob("data/policies/*.md")):
        text = open(path).read()
        chunks = chunk_text(text)
        embeddings = model.encode(chunks)
        for i, (c, e) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                "INSERT INTO policy_chunks (source, chunk_index, content, embedding) VALUES (%s,%s,%s,%s)",
                (os.path.basename(path), i, c, e),
            )
        total += len(chunks)
        print(f"{path}: {len(chunks)} chunks")
    print("Total chunks:", total)
