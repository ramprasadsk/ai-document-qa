
from fastapi.testclient import TestClient
import os

os.environ["BACKEND_API_KEY"] = "local-test-key"
from backend.main import app

from unittest.mock import patch
from backend.rag_service import RAGResponse
from fastapi.testclient import TestClient
from backend.main import app
import hashlib

TEST_SESSION_TOKEN = "test-session-token-for-automated-tests-12345"

client = TestClient(
    app,
    headers={
        "X-API-Key": "local-test-key",
        "X-Session-Token": TEST_SESSION_TOKEN
    }
)


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
            "ids": ["test_document_0"],
            "metadatas": [
                {
                    "session_hash": hashlib.sha256(
                        TEST_SESSION_TOKEN.encode()
                    ).hexdigest()
                }
            ]
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

    with TestClient(
        app,
        headers={
            "X-API-Key": "local-test-key",
            "X-Session-Token": TEST_SESSION_TOKEN
        }
    ) as test_client:
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

def test_missing_api_key():
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        response = test_client.post(
            "/questions/ask",
            json={
                "document_id": "nonexistent_document",
                "question": "Test question"
            }
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."

def test_invalid_api_key():
    from fastapi.testclient import TestClient

    with TestClient(
        app,
        headers={"X-API-Key": "wrong-api-key"}
    ) as test_client:
        response = test_client.post(
            "/questions/ask",
            json={
                "document_id": "nonexistent_document",
                "question": "Test question"
            }
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."

def test_user_cannot_access_another_sessions_document():
    import hashlib
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    user_a_token = "user-a-private-session-token-123456789"
    user_b_token = "user-b-private-session-token-987654321"

    user_a_hash = hashlib.sha256(
        user_a_token.encode()
    ).hexdigest()

    with patch("backend.main.collection.get") as mock_get, \
         patch("backend.main.retrieve_chunks") as mock_retrieve, \
         patch("backend.main.generate_answer") as mock_generate:

        # Simulate a document belonging to User A.
        mock_get.return_value = {
            "ids": ["private_document_0"],
            "metadatas": [
                {"session_hash": user_a_hash}
            ]
        }

        # User B attempts to access User A's document.
        with TestClient(
            app,
            headers={
                "X-API-Key": "local-test-key",
                "X-Session-Token": user_b_token
            },
            client=("192.0.2.20", 50000)
        ) as test_client:

            response = test_client.post(
                "/questions/ask",
                json={
                    "document_id": "private_document",
                    "question": "What is in this PDF?"
                }
            )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "Access denied to this document."
        )

        # Unauthorized requests must never reach retrieval or Gemini.
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()

def test_same_pdf_has_different_ids_for_different_sessions():
    import hashlib
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    pdf_bytes = b"%PDF-1.4\nTest PDF content"

    user_a_token = "user-a-session-token-123456789012345"
    user_b_token = "user-b-session-token-987654321098765"

    with patch("backend.main.collection.get") as mock_get, \
         patch("backend.main.extract_chunks") as mock_extract, \
         patch("backend.main.embed_chunks") as mock_embed, \
         patch("backend.main.store_chunks_in_chroma") as mock_store:

        mock_get.return_value = {"ids": []}
        mock_extract.return_value = [
            {"text": "Sample PDF text", "page": 1}
        ]
        mock_embed.return_value = [
            {
                "text": "Sample PDF text",
                "page": 1,
                "embedding": [0.1, 0.2, 0.3]
            }
        ]

        def upload_as(session_token, ip):
            with TestClient(
                app,
                headers={
                    "X-API-Key": "local-test-key",
                    "X-Session-Token": session_token
                },
                client=(ip, 50000)
            ) as test_client:
                return test_client.post(
                    "/documents/upload",
                    files={
                        "file": (
                            "sample.pdf",
                            pdf_bytes,
                            "application/pdf"
                        )
                    }
                )

        response_a = upload_as(user_a_token, "192.0.2.30")
        response_b = upload_as(user_b_token, "192.0.2.31")

        assert response_a.status_code == 200
        assert response_b.status_code == 200

        document_a = response_a.json()["document_id"]
        document_b = response_b.json()["document_id"]

        assert document_a != document_b

        # Verify both documents were stored with their owners.
        assert mock_store.call_count == 2

        stored_owners = [
            call.args[2]
            for call in mock_store.call_args_list
        ]

        assert hashlib.sha256(
            user_a_token.encode()
        ).hexdigest() in stored_owners

        assert hashlib.sha256(
            user_b_token.encode()
        ).hexdigest() in stored_owners