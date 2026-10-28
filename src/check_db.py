import psycopg

conn = psycopg.connect("postgresql://postgres:dev@localhost:5432/postgres")
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()
cur.execute("SELECT version();")
print(cur.fetchone()[0])
print("Database + vector extension OK")
conn.close()