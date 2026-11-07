import os, re
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from metadata import query_allowed_sources

DB_URL = os.getenv("DB_URL", "postgresql://postgres:dev@localhost:5432/postgres")
_emb = SentenceTransformer("all-MiniLM-L6-v2")
_ce = None

def tok(s):
    return re.findall(r"\w+", s.lower())

class Retriever:
    def __init__(self):
        self.conn = psycopg.connect(DB_URL)
        register_vector(self.conn)
        self.rows = self.conn.execute(
            "SELECT id, source, content FROM policy_chunks ORDER BY id").fetchall()
        self.by_id = {r[0]: r for r in self.rows}
        self.sources = sorted({r[1] for r in self.rows})
        self.bm25 = BM25Okapi([tok(r[2]) for r in self.rows])

    def vector(self, q, n, allowed=None):
        e = _emb.encode(q)
        if allowed is None:
            return self.conn.execute(
                "SELECT id, source, content FROM policy_chunks ORDER BY embedding <=> %s LIMIT %s",
                (e, n)).fetchall()
        return self.conn.execute(
            "SELECT id, source, content FROM policy_chunks WHERE source = ANY(%s) "
            "ORDER BY embedding <=> %s LIMIT %s", (allowed, e, n)).fetchall()

    def keyword(self, q, n, allowed=None):
        scores = self.bm25.get_scores(tok(q))
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = [self.rows[i] for i in idx if allowed is None or self.rows[i][1] in allowed]
        return out[:n]

    def hybrid(self, q, n, allowed=None, k=60):
        fused = {}
        for lst in (self.vector(q, 20, allowed), self.keyword(q, 20, allowed)):
            for rank, r in enumerate(lst):
                fused[r[0]] = fused.get(r[0], 0) + 1 / (k + rank + 1)
        ids = sorted(fused, key=lambda i: -fused[i])[:n]
        return [self.by_id[i] for i in ids]

    def rerank(self, q, n, allowed=None, pool=10):
        global _ce
        if _ce is None:
            _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        cands = self.hybrid(q, pool, allowed)
        scores = _ce.predict([(q, c[2]) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])[:n]
        return [cands[i] for i in order]

    def search(self, q, mode, n):
        allowed = None
        if os.getenv("FILTER") == "1":
            allowed = query_allowed_sources(q, self.sources)
        fn = {"vector": self.vector, "bm25": self.keyword,
              "hybrid": self.hybrid, "rerank": self.rerank}[mode]
        return fn(q, n, allowed)
