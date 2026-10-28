from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("What happens if I miss an EMI payment?")
print("Vector length:", len(vec))
print("First 5 numbers:", vec[:5])