
from fastapi.testclient import TestClient
from backend.main import app

from unittest.mock import patch
from backend.rag_service import RAGResponse


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ai-document-qa"
    }


def test_reject_fake_pdf():
    fake_pdf = b"This is not a real PDF file."

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "fake.pdf",
                fake_pdf,
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400


def test_question_with_unknown_document():
    response = client.post(
        "/questions/ask",
        json={
            "document_id": "nonexistent_document_12345",
            "question": "What is CAN SLIM?"
        }
    )

    assert response.status_code == 404

def test_successful_question_answering():
    fake_chunks = [
        {
            "page": 11,
            "text": "CAN SLIM is a stock selection strategy."
        }
    ]

    fake_answer = RAGResponse(
        supported=True,
        answer="CAN SLIM is a stock selection strategy.",
        source_pages=[11]
    )

    with patch("backend.main.collection.get") as mock_get, \
         patch("backend.main.retrieve_chunks") as mock_retrieve, \
         patch("backend.main.generate_answer") as mock_generate:

        mock_get.return_value = {
            "ids": ["test_document_0"]
        }

        mock_retrieve.return_value = (
            fake_chunks,
            fake_chunks
        )

        mock_generate.return_value = fake_answer

        response = client.post(
            "/questions/ask",
            json={
                "document_id": "test_document",
                "question": "What is CAN SLIM?"
            }
        )

    assert response.status_code == 200

    assert response.json() == {
        "supported": True,
        "answer": "CAN SLIM is a stock selection strategy.",
        "source_pages": [11]
    }

    mock_retrieve.assert_called_once_with(
        "What is CAN SLIM?",
        "test_document"
    )

    mock_generate.assert_called_once_with(
        "What is CAN SLIM?",
        fake_chunks
    )

def test_question_rate_limit():
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        responses = [
            test_client.post(
                "/questions/ask",
                json={
                    "document_id": "nonexistent_document",
                    "question": "Test question"
                }
            )
            for _ in range(6)
        ]

    assert responses[-1].status_code == 429