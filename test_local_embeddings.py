
from sentence_transformers import SentenceTransformer

# Download on first run, then load from local cache
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Successful companies often show strong earnings growth.",
    "Businesses with rising profits may see their stock prices increase.",
    "How do I bake a chocolate cake?"
]

embeddings = model.encode(sentences)

print("Number of sentences:", len(embeddings))
print("Vector dimensions:", len(embeddings[0]))
print("First embedding preview:", embeddings[0][:5])
