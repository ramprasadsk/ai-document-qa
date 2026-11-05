
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

sentences = [
    "A company is reporting increasing profits.",
    "The business's earnings are growing.",
    "How do I bake a chocolate cake?"
]

for sentence in sentences:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence
    )

    vector = result.embeddings[0].values

    print("\nSentence:", sentence)
    print("Vector dimensions:", len(vector))
    print("First 10 numbers:", vector[:10])
