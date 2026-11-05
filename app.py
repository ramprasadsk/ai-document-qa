
import os
import math
import hashlib

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

ANSWER_MODEL = "gemini-3.5-flash-lite"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.35
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Document Q&A")
st.write("Upload one PDF and ask questions based on its content.")


@st.cache_resource
def load_embedding_model():
    """Load the local embedding model once and reuse it."""
    return SentenceTransformer(LOCAL_EMBEDDING_MODEL)


embedding_model = load_embedding_model()


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


def cosine_similarity(vector_a, vector_b):
    """Measure how similar two vectors are."""
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_chunks(question, embedded_chunks):
    """Find the document chunks most relevant to the question."""
    question_vector = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    ranked_chunks = []

    for chunk in embedded_chunks:
        score = cosine_similarity(
            question_vector,
            chunk["embedding"]
        )

        ranked_chunks.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": score
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


def generate_answer(question, relevant_chunks):
    """Generate an answer using retrieved PDF evidence."""
    context = "\n\n".join(
        f"[PDF page {chunk['page']}]\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
You are a helpful study assistant.

Answer the student's question using only the PDF evidence below.

- Do not use outside knowledge.
- Do not invent facts.
- Explain the answer in clear, simple language.
- If the evidence is insufficient, say so clearly.
- Include relevant PDF page numbers.
- Answer the question actually asked.

Student's question:
{question}

PDF evidence:
{context}
"""

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt
    )

    return response.text or "The model returned an empty answer."


uploaded_pdf = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_id = hashlib.sha256(pdf_bytes).hexdigest()

    # Reset the conversation when a different PDF is uploaded.
    if st.session_state.get("pdf_id") != pdf_id:
        st.session_state["pdf_id"] = pdf_id
        st.session_state["embedded_chunks"] = None
        st.session_state["messages"] = []

    if st.session_state.get("embedded_chunks") is None:
        try:
            with st.spinner("Reading your PDF..."):
                chunks = extract_chunks(pdf_bytes)

            if not chunks:
                st.error(
                    "No readable text was found. "
                    "This PDF may contain scanned images."
                )
                st.stop()

            st.info(f"Extracted {len(chunks)} text chunks.")

            with st.spinner("Preparing your document locally..."):
                embedded_chunks = embed_chunks(chunks)

            st.session_state["embedded_chunks"] = embedded_chunks
            st.success("Your PDF is ready!")

        except Exception as error:
            st.error(f"Could not process the PDF: {error}")
            st.stop()

    st.caption(f"Document: {uploaded_pdf.name}")

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your PDF..."):
                    relevant_chunks = retrieve_chunks(
                        question,
                        st.session_state["embedded_chunks"]
                    )

                if not relevant_chunks:
                    answer = (
                        "I couldn't find sufficiently relevant "
                        "information in this PDF to answer confidently."
                    )
                else:
                    with st.spinner("Generating your answer..."):
                        answer = generate_answer(
                            question,
                            relevant_chunks
                        )

                st.markdown(answer)

                if relevant_chunks:
                    with st.expander("View sources used"):
                        for index, chunk in enumerate(
                            relevant_chunks,
                            start=1
                        ):
                            st.markdown(
                                f"**Source {index} — PDF page "
                                f"{chunk['page']}**"
                            )

                            st.caption(
                                f"Similarity score: {chunk['score']:.3f}"
                            )

                            st.write(chunk["text"])
                            st.divider()

            except Exception as error:
                answer = f"Sorry, I couldn't process your question: {error}"
                st.error(answer)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer
        })
