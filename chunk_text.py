
import re
import pymupdf

PDF_PATH = "documents/How-To-Make-Money.pdf"
CHUNK_SIZE = 1000
OVERLAP_SIZE = 150


def clean_page_text(text):
    # Join words split by a hyphen at a line break
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Replace remaining line breaks with spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_sentences(text):
    # Simple sentence-boundary approximation
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_page(text, page_number):
    sentences = split_into_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        # Handle unusually long sentences
        if len(sentence) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            while start < len(sentence):
                piece = sentence[start:start + CHUNK_SIZE]
                chunks.append(piece)
                start += CHUNK_SIZE - OVERLAP_SIZE

            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)

            # Reuse the end of the previous chunk
            overlap = current[-OVERLAP_SIZE:] if current else ""
            current = f"{overlap} {sentence}".strip()

    if current:
        chunks.append(current)

    return [
        {"page": page_number, "text": chunk}
        for chunk in chunks
        if chunk.strip()
    ]


all_chunks = []

with pymupdf.open(PDF_PATH) as doc:
    for index, page in enumerate(doc):
        raw_text = page.get_text()
        text = clean_page_text(raw_text)

        if not text:
            continue

        page_chunks = chunk_page(text, index + 1)
        all_chunks.extend(page_chunks)

print("Total chunks:", len(all_chunks))

for i, chunk in enumerate(all_chunks[:5]):
    print(f"\n--- Chunk {i + 1} | Page {chunk['page']} ---")
    print(chunk["text"])

import json

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, ensure_ascii=False, indent=2)

print("\nSaved chunks to chunks.json")