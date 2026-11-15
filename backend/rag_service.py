import chromadb
import os
import re

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field

import pymupdf
from pathlib import Path

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
SIMILARITY_THRESHOLD = 0.35

CANDIDATE_K = 20
FINAL_K = 7
RRF_K = 60

SEMANTIC_CANDIDATE_K = 20
KEYWORD_CANDIDATE_K = 20
FINAL_K = 5
MAX_CONTEXT_CHUNKS = 8
RRF_K = 60

MIN_SEMANTIC_SCORE = 0.25
MIN_KEYWORD_SCORE = 1.0

COLLECTION_NAME = "pdf_chunks"
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
ANSWER_MODEL = "gemini-3.5-flash-lite"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STOPWORDS = {
    "what", "where", "when", "why", "how",
    "is", "are", "was", "were", "the",
    "a", "an", "of", "in", "on", "to",
    "for", "and", "or", "does", "do",
    "did", "each", "which", "who",
    "according", "explain", "describe",
    "stand"
}
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=api_key)

def load_embedding_model():
    """Load the local embedding model once and reuse it."""
    return SentenceTransformer(LOCAL_EMBEDDING_MODEL)

def load_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

chroma_client = load_chroma_client()
embedding_model = load_embedding_model()

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def extract_chunks(pdf_bytes):
    """Extract PDF text and split it into overlapping chunks."""
    chunks = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:
        for page_number, page in enumerate(document, start=1):
            text = " ".join(page.get_text("text").split())

            start = 0

            while start < len(text):
                end = start + CHUNK_SIZE
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "page": page_number,
                        "text": chunk_text
                    })

                if end >= len(text):
                    break

                start = end - CHUNK_OVERLAP

    return chunks



def embed_chunks(chunks):
    """Create embeddings locally for document chunks."""
    embedded_chunks = []

    for start in range(0, len(chunks), 40):
        batch = chunks[start:start + 40]
        texts = [chunk["text"] for chunk in batch]

        vectors = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        for chunk, vector in zip(batch, vectors):
            embedded_chunks.append({
                "page": chunk["page"],
                "text": chunk["text"],
                "embedding": vector.tolist()
            })

    return embedded_chunks

def store_chunks_in_chroma(pdf_id, embedded_chunks, session_hash):
    """Store PDF chunks and their embeddings in ChromaDB."""

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, chunk in enumerate(embedded_chunks):
        ids.append(f"{pdf_id}_{index}")
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"])

        metadatas.append({
            "pdf_id": pdf_id,
            "page": chunk["page"],
            "session_hash": session_hash
        })

    for start in range(0, len(ids), 100):
        end = start + 100

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end]
        )

def tokenize(text):
    """Tokenize text and preserve hyphenated acronyms."""

    # Normalize variations of CAN SLIM into one searchable term.
    text = re.sub(
        r"\bC[\s-]*A[\s-]*N[\s-]*S[\s-]*L[\s-]*I[\s-]*M\b",
        "canslim",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bCAN[\s-]+SLIM\b",
        "canslim",
        text,
        flags=re.IGNORECASE
    )

    words = re.findall(r"[a-z0-9]+", text.lower())

    return [
        word for word in words
        if word not in STOPWORDS
    ]

def normalize_text(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())

def retrieve_chunks(question, pdf_id):
    """Hybrid retrieval with evidence filtering and neighboring chunks."""

    stored = collection.get(
        where={"pdf_id": pdf_id},
        include=["documents", "metadatas"]
    )

    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]

    if not ids:
        return [], []

    chunk_lookup = {
        chunk_id: {
            "id": chunk_id,
            "page": metadata["page"],
            "text": text
        }
        for chunk_id, text, metadata in zip(
            ids, documents, metadatas
        )
    }

    # 1. Semantic search
    question_vector = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(SEMANTIC_CANDIDATE_K, len(ids)),
        where={"pdf_id": pdf_id},
        include=["distances"]
    )

    semantic_ids = semantic_results["ids"][0]

    semantic_scores = {
        chunk_id: 1 - distance
        for chunk_id, distance in zip(
            semantic_ids,
            semantic_results["distances"][0]
        )
    }

    # 2. BM25 keyword search
    tokenized_corpus = [
        tokenize(text)
        for text in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(question)

    if query_tokens:
        keyword_scores = bm25.get_scores(query_tokens)

        keyword_ranked = sorted(
            range(len(ids)),
            key=lambda i: keyword_scores[i],
            reverse=True
        )

        keyword_ids = [
            ids[i]
            for i in keyword_ranked
            if keyword_scores[i] >= MIN_KEYWORD_SCORE
        ][:KEYWORD_CANDIDATE_K]

    else:
        keyword_ids = []

    # 3. Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, chunk_id in enumerate(semantic_ids, 1):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

    for rank, chunk_id in enumerate(keyword_ids, 1):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    # 4. Filter weak evidence
    accepted_ids = []

    for chunk_id in ranked_ids:
        semantic_score = semantic_scores.get(chunk_id)

        semantic_match = (
            semantic_score is not None
            and semantic_score >= MIN_SEMANTIC_SCORE
        )

        keyword_match = chunk_id in keyword_ids

        if semantic_match or keyword_match:
            accepted_ids.append(chunk_id)

        if len(accepted_ids) >= FINAL_K:
            break

    # 5. Expand with neighboring chunks
    selected_ids = []
    seen = set()

    def add_chunk(chunk_id):
        if chunk_id in chunk_lookup and chunk_id not in seen:
            seen.add(chunk_id)
            selected_ids.append(chunk_id)

    for chunk_id in accepted_ids:
        add_chunk(chunk_id)

    # IDs were generated as: <pdf_id>_<chunk_index>
    # Retrieve the immediately following chunk.
    for chunk_id in accepted_ids:
        if len(selected_ids) >= MAX_CONTEXT_CHUNKS:
            break

        chunk_index = int(chunk_id.rsplit("_", 1)[1])
        next_id = f"{pdf_id}_{chunk_index + 1}"

        add_chunk(next_id)

    # 6. Prepare results
    top_chunks = []

    for chunk_id in selected_ids:
        chunk = chunk_lookup[chunk_id]

        semantic_score = semantic_scores.get(chunk_id)

        if chunk_id not in accepted_ids:
            method = "Neighbor"
        elif chunk_id in keyword_ids and chunk_id in semantic_ids:
            method = "Semantic + Keyword"
        elif chunk_id in keyword_ids:
            method = "Keyword"
        else:
            method = "Semantic"

        top_chunks.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": semantic_score,
            "rrf_score": rrf_scores.get(chunk_id),
            "retrieval_method": method
        })

    return top_chunks, top_chunks

class RAGResponse(BaseModel):
    supported: bool = Field(
        description="Whether the PDF evidence sufficiently supports the answer"
    )

    answer: str = Field(
        description="Answer based only on the retrieved PDF evidence"
    )

    source_pages: list[int] = Field(
        description="PDF page numbers that directly support the answer"
    )

def generate_answer(question, relevant_chunks):
    """Generate a structured answer grounded in PDF evidence."""

    context = "\n\n".join(
        f"[PDF page {chunk['page']}]\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
You are a study assistant answering questions from an uploaded PDF.

Use ONLY the PDF evidence provided below.

Rules:
1. Answer only when the evidence supports the answer.
2. Do not use outside knowledge or invent missing details.
3. If the evidence is insufficient, set supported to false.
4. If only part of a multi-part question can be answered, set supported
   to false rather than inventing the missing parts.
5. When supported is false, explain briefly what evidence is missing.
6. When supported is true, give a clear, complete answer.
7. source_pages must contain only PDF page numbers explicitly supplied
   in the evidence and directly supporting the answer.
8. Do not confuse printed book page numbers with PDF page numbers.
9. Before answering, identify all parts of the
   student's question that the evidence addresses.
10. Include every essential supported point,
    especially when the question asks for a list,
    characteristics, reasons, or requirements.
11. Do not omit a relevant point merely to make
    the answer shorter.
12. Do not introduce details that the supplied
    evidence does not support.
13. When answering an acronym or multi-part question,
    check every requested component against the evidence.
    If all components are supported, set supported to true.
    Do not reject an answer merely because the information
    appears across multiple retrieved chunks or pages.

Student's question:
{question}

PDF evidence:
{context}
"""

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RAGResponse,
            "temperature": 0
        }
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    result = RAGResponse.model_validate_json(response.text)

    # Validate citations against pages actually retrieved.
    retrieved_pages = {
        chunk["page"] for chunk in relevant_chunks
    }

    if not result.supported:
        return RAGResponse(
            supported=False,
            answer=(
                "I couldn't find sufficient evidence in this PDF "
                "to answer the question confidently."
            ),
            source_pages=[]
        )

    if not result.answer.strip():
        raise ValueError("Gemini returned an empty answer.")

    valid_pages = sorted(
        set(result.source_pages) & retrieved_pages
    )

    # A supported answer must have at least one valid citation.
    if not valid_pages:
        return RAGResponse(
            supported=False,
            answer=(
                "I couldn't verify the answer against "
                "the retrieved PDF pages."
            ),
            source_pages=[]
        )

    return RAGResponse(
        supported=True,
        answer=result.answer,
        source_pages=valid_pages
    )