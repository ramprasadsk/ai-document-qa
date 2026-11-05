
from google import genai
from dotenv import load_dotenv
import os
import math

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

sentences = {
    "A": "A company is reporting increasing profits.",
    "B": "The business's earnings are growing.",
    "C": "How do I bake a chocolate cake?"
}

# Step 1: Convert each sentence into an embedding vector
vectors = {}

for name, sentence in sentences.items():
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence
    )

    vectors[name] = result.embeddings[0].values


# Step 2: Calculate cosine similarity between two vectors
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    return dot_product / (magnitude_a * magnitude_b)


# Step 3: Compare the sentences
pairs = [("A", "B"), ("A", "C"), ("B", "C")]

for first, second in pairs:
    score = cosine_similarity(vectors[first], vectors[second])

    print(f"{first} vs {second}: {score:.4f}")
    print(f"  {sentences[first]}")
    print(f"  {sentences[second]}")
    print()
