
import json
import math

from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-embedding-001"

# Load chunks and their precomputed embeddings
with open("embedded_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Ask a question
question = input("Ask a question about the PDF: ")

# Embed the question using the same dimensions as the chunks
result = client.models.embed_content(
    model=MODEL,
    contents=question,
    config={"output_dimensionality": 768}
)

question_vector = result.embeddings[0].values

# Compare the question against every chunk
ranked_chunks = []

for chunk in chunks:
    score = cosine_similarity(
        question_vector,
        chunk["embedding"]
    )

    ranked_chunks.append({
        "score": score,
        "page": chunk["page"],
        "text": chunk["text"]
    })

# Highest similarity first
ranked_chunks.sort(
    key=lambda item: item["score"],
    reverse=True
)

# Display only chunks that pass an initial threshold
TOP_K = 5
SIMILARITY_THRESHOLD = 0.60

matching_chunks = [
    chunk for chunk in ranked_chunks
    if chunk["score"] >= SIMILARITY_THRESHOLD
]

if not matching_chunks:
    print("\nNo sufficiently relevant chunks found.")
else:
    print(f"\nTop {min(TOP_K, len(matching_chunks))} matching chunks:")

    for rank, chunk in enumerate(matching_chunks[:TOP_K], start=1):
        print(f"\n--- Rank {rank} ---")
        print(f"Similarity: {chunk['score']:.4f}")
        print(f"PDF page: {chunk['page']}")
        print(f"Text: {chunk['text']}")
