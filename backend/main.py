
import hashlib
from pydantic import BaseModel, Field
from backend.rag_service import (
    collection,
    extract_chunks,
    embed_chunks,
    store_chunks_in_chroma,
    retrieve_chunks,
    generate_answer,
    RAGResponse,
)
from fastapi import FastAPI, File, HTTPException, UploadFile
from google.genai.errors import ServerError, ClientError

app = FastAPI(
    title="AI Document Q&A API",
    description="Backend API for our PDF-based RAG application",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-document-qa"
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Limit uploads to 10 MB.
    pdf_bytes = await file.read(10 * 1024 * 1024 + 1)

    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB upload limit."
        )

    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid or empty PDF file."
        )

    document_id = hashlib.sha256(pdf_bytes).hexdigest()

    # Check whether this PDF was previously indexed.
    existing = collection.get(
        where={"pdf_id": document_id},
        limit=1
    )

    if existing["ids"]:
        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "already_indexed"
        }

    # Extract text and create chunks.
    try:
        chunks = extract_chunks(pdf_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF."
        )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text."
        )

    # Generate local embeddings and store in ChromaDB.
    embedded_chunks = embed_chunks(chunks)
    store_chunks_in_chroma(document_id, embedded_chunks)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "status": "indexed"
    }


@app.post("/questions/ask", response_model=RAGResponse)
def ask_question(request: QuestionRequest):

    # Verify that the requested document exists.
    existing = collection.get(
        where={"pdf_id": request.document_id},
        limit=1
    )

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload the PDF first."
        )

    # Retrieve relevant evidence from the PDF.
    relevant_chunks, _ = retrieve_chunks(
        request.question,
        request.document_id
    )

    # Reject questions without sufficient retrieved evidence.
    if not relevant_chunks:
        return RAGResponse(
            supported=False,
            answer="I couldn't find relevant evidence in this PDF.",
            source_pages=[]
        )

    # Generate a grounded answer using Gemini.
    try:
        return generate_answer(
            request.question,
            relevant_chunks
        )

    except ServerError as exc:
        if exc.code == 503:
            raise HTTPException(
                status_code=503,
                detail="The AI model is temporarily busy. Please try again shortly."
            ) from exc
        raise

    except ClientError as exc:
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="The AI service rate limit has been reached. Please try again later."
            ) from exc
        raise