
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-embedding-001"
BATCH_SIZE = 40
OUTPUT_FILE = "embedded_chunks.json"

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Resume from embeddings already saved
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        embedded_chunks = json.load(file)
else:
    embedded_chunks = []

completed = len(embedded_chunks)

if completed > len(chunks):
    raise ValueError("Saved embeddings exceed the number of chunks.")

print(f"Total chunks: {len(chunks)}")
print(f"Already embedded: {completed}")

while completed < len(chunks):
    batch = chunks[completed:completed + BATCH_SIZE]
    texts = [chunk["text"] for chunk in batch]

    try:
        result = client.models.embed_content(
            model=MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )

        if len(result.embeddings) != len(batch):
            raise RuntimeError("Embedding count does not match batch size.")

        for chunk, embedding in zip(batch, result.embeddings):
            embedded_chunks.append({
                "page": chunk["page"],
                "text": chunk["text"],
                "embedding": embedding.values
            })

        completed = len(embedded_chunks)

        # Save after each successful batch
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(embedded_chunks, file)

        print(f"Embedded {completed} / {len(chunks)} chunks")

        # Pause to reduce the chance of hitting rate limits
        if completed < len(chunks):
            time.sleep(30)

    except APIError as error:
        if error.code == 429:
            print("Rate limit reached. Waiting 65 seconds...")
            time.sleep(65)
        else:
            raise

print("\nAll embeddings saved successfully.")
print("Vector dimensions:", len(embedded_chunks[0]["embedding"]))
