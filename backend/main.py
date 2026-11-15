
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
from google.genai.errors import ServerError, ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Depends, Security, Header
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import os
import secrets

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

app = FastAPI(
    title="AI Document Q&A API",
    description="Backend API for our PDF-based RAG application",
    version="1.0.0"
)

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

class QuestionRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    provided_key: str | None = Security(api_key_header)
):
    if not BACKEND_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Backend API authentication is not configured."
        )

    if not provided_key or not secrets.compare_digest(
        provided_key,
        BACKEND_API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )

def verify_session_token(
    session_token: str | None = Header(
        default=None,
        alias="X-Session-Token"
    )
) -> str:
    if not session_token:
        raise HTTPException(
            status_code=401,
            detail="Missing session token."
        )

    if len(session_token) < 32 or len(session_token) > 128:
        raise HTTPException(
            status_code=401,
            detail="Invalid session token."
        )

    return session_token

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-document-qa"
    }

@app.post(
    "/documents/upload",
    dependencies=[
        Depends(verify_api_key),
        Depends(verify_session_token)
    ]
)
@limiter.limit("2/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_token: str = Depends(verify_session_token)
):
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

    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    session_hash = hashlib.sha256(session_token.encode()).hexdigest()

    document_id = hashlib.sha256(
        f"{session_hash}:{pdf_hash}".encode()
    ).hexdigest()

    # Check whether this PDF was previously indexed.
    existing = collection.get(
        where={"pdf_id": document_id},
        limit=1,
        include=["metadatas"]
    )

    if existing["ids"]:
        stored_session_hash = existing["metadatas"][0].get(
            "session_hash"
        )

        if not stored_session_hash or not secrets.compare_digest(
            stored_session_hash,
            session_hash
        ):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document."
            )

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
    store_chunks_in_chroma(
        document_id,
        embedded_chunks,
        session_hash
    )
    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "status": "indexed"
    }


@app.post(
    "/questions/ask",
    response_model=RAGResponse,
    dependencies=[
        Depends(verify_api_key),
        Depends(verify_session_token)
    ]
)
@limiter.limit("5/minute")
def ask_question(
    request: Request,
    question_data: QuestionRequest,
    session_token: str = Depends(verify_session_token)
):
    session_hash = hashlib.sha256(
        session_token.encode()
    ).hexdigest()

    existing = collection.get(
        where={"pdf_id": question_data.document_id},
        limit=1,
        include=["metadatas"]
    )

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    stored_session_hash = existing["metadatas"][0].get(
        "session_hash"
    )

    if not stored_session_hash or not secrets.compare_digest(
        stored_session_hash,
        session_hash
    ):
        raise HTTPException(
            status_code=403,
            detail="Access denied to this document."
        )

    # Retrieve relevant evidence from the PDF.
    relevant_chunks, _ = retrieve_chunks(
        question_data.question,
        question_data.document_id
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
            question_data.question,
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