import os, glob
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv("DB_URL", "postgresql://postgres:dev@localhost:5432/postgres")
CHUNKER = os.getenv("CHUNKER", "paragraph")   # fixed | paragraph | line
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_fixed(text, size=400, overlap=80):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]

def split_heading(text):
    lines = text.strip().split("\n")
    heading = lines[0] if lines[0].startswith("#") else ""
    body = "\n".join(lines[1:] if heading else lines)
    return heading, body

def chunk_paragraph(text, max_chars=400):
    heading, body = split_heading(text)
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return [f"{heading}\n\n{c}" if heading else c for c in chunks]

def chunk_line(text):
    heading, body = split_heading(text)
    h = heading.lstrip("# ").strip()
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    return [f"{h}: {l}" if h else l for l in lines]

CHUNKERS = {"fixed": chunk_fixed, "paragraph": chunk_paragraph, "line": chunk_line}
chunk_text = CHUNKERS[CHUNKER]

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
        chunks = chunk_text(open(path).read())
        embeddings = model.encode(chunks)
        for i, (c, e) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                "INSERT INTO policy_chunks (source, chunk_index, content, embedding) VALUES (%s,%s,%s,%s)",
                (os.path.basename(path), i, c, e),
            )
        total += len(chunks)
    print(f"[{CHUNKER}] total chunks: {total}")
