
import json
import math
import os

from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
ANSWER_MODEL = "gemini-3.5-flash-lite"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.60


# Load the PDF chunks and their previously generated embeddings
with open("embedded_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Calculate cosine similarity between two vectors
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Retrieve the most relevant chunks from the PDF
def retrieve_relevant_chunks(question):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config={"output_dimensionality": 768}
    )

    question_vector = result.embeddings[0].values

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

    ranked_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    matching_chunks = [
        chunk for chunk in ranked_chunks
        if chunk["score"] >= SIMILARITY_THRESHOLD
    ]

    return matching_chunks[:TOP_K]


# Generate an answer using only the retrieved PDF evidence
def generate_answer(question, relevant_chunks):
    context = ""

    for chunk in relevant_chunks:
        context += (
            f"\n[PDF page {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are a helpful study assistant.

Answer the student's question using ONLY the PDF evidence
provided below.

Rules:
1. Do not use outside knowledge to answer the question.
2. Do not invent facts or details that are not in the evidence.
3. Answer in clear, simple language.
4. If the evidence does not contain enough information,
   clearly say that the PDF does not provide enough information.
5. Answer the question that was actually asked.
6. Include the relevant PDF page number(s) in your answer.
7. Summarize or explain the evidence in your own words.
   Do not copy large passages unnecessarily.

Student's question:
{question}

PDF evidence:
{context}
"""

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt
    )

    return response.text


# Main program
question = input("\nAsk a question about your PDF: ").strip()

if not question:
    print("Please enter a question.")
else:
    relevant_chunks = retrieve_relevant_chunks(question)

    if not relevant_chunks:
        print(
            "\nI couldn't find sufficiently relevant information "
            "in this PDF to answer confidently."
        )
    else:
        print("\nRetrieved evidence:")

        for index, chunk in enumerate(relevant_chunks, start=1):
            print(
                f"\n{index}. Page {chunk['page']} "
                f"(similarity: {chunk['score']:.4f})"
            )
            print(chunk["text"][:300].replace("\n", " "))
            print("...")

        print("\nGenerating answer...")

        answer = generate_answer(question, relevant_chunks)

        print("\nAnswer:")
        print(answer)
