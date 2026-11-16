import os
import hashlib
import requests
import streamlit as st
import secrets

from dotenv import load_dotenv

load_dotenv()

def get_config(key, default=""):
    try:
        return st.secrets.get(
            key,
            os.getenv(key, default)
        )
    except FileNotFoundError:
        return os.getenv(key, default)

BACKEND_URL = get_config(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)

BACKEND_API_KEY = get_config("BACKEND_API_KEY")

API_HEADERS = {
    "X-API-Key": BACKEND_API_KEY
}

if "session_token" not in st.session_state:
    st.session_state["session_token"] = secrets.token_urlsafe(32)

SESSION_HEADERS = {
    **API_HEADERS,
    "X-Session-Token": st.session_state["session_token"]
}

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Document Q&A")
st.write("Upload one PDF and ask questions based on its content.")

def delete_pdf_from_backend(document_id):
    """Delete the current session's PDF from ChromaDB."""

    response = requests.delete(
        f"{BACKEND_URL}/documents/{document_id}",
        headers=SESSION_HEADERS,
        timeout=60
    )

    response.raise_for_status()
    return response.json()

def upload_pdf_to_backend(uploaded_file):
    """Send the uploaded PDF to FastAPI for indexing."""

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    response = requests.post(
        f"{BACKEND_URL}/documents/upload",
        files=files,
        headers=SESSION_HEADERS,
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def ask_backend(question, document_id):
    """Send a question to FastAPI and return its answer."""

    response = requests.post(
        f"{BACKEND_URL}/questions/ask",
        json={
            "document_id": document_id,
            "question": question
        },
        headers=SESSION_HEADERS,
        timeout=120
    )

    response.raise_for_status()

    return response.json()

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded_pdf = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    key=f"pdf_uploader_{st.session_state['uploader_key']}"
)

if st.session_state.get("document_indexed") is True:
    if st.button("🗑️ Delete Document", type="secondary"):
        try:
            with st.spinner("Deleting document..."):
                delete_pdf_from_backend(
                    st.session_state["document_id"]
                )

            st.session_state["document_indexed"] = None
            st.session_state["document_id"] = None
            st.session_state["pdf_hash"] = None
            st.session_state["messages"] = []
            st.session_state["document_deleted"] = False

            # Reset the uploader so the same PDF can be selected again.
            st.session_state["uploader_key"] += 1

            st.rerun()

        except requests.exceptions.RequestException as error:
            st.error(f"Could not delete document: {error}")

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Reset the conversation when a different PDF is uploaded.
    if st.session_state.get("pdf_hash") != pdf_hash:
        st.session_state["pdf_hash"] = pdf_hash
        st.session_state["document_indexed"] = None
        st.session_state["document_id"] = None
        st.session_state["messages"] = []
        st.session_state["document_deleted"] = False

    
    if (
        st.session_state.get("document_indexed") is None
        and not st.session_state.get("document_deleted", False)
    ):
        try:
            with st.spinner("Uploading and indexing PDF through FastAPI..."):
                result = upload_pdf_to_backend(uploaded_pdf)

            if result["status"] == "already_indexed":
                st.success("PDF already indexed. Reusing stored vectors!")
            elif result["status"] == "indexed":
                st.success(
                    f"PDF indexed successfully! "
                    f"{result['chunks']} chunks stored."
                )
            # Store the session-specific document ID returned by FastAPI
            st.session_state["document_id"] = result["document_id"]

            st.session_state["document_indexed"] = True

        except requests.exceptions.RequestException as error:
            st.error(f"FastAPI backend error: {error}")
            st.stop()

    if st.session_state.get("document_deleted", False):
        st.info("Document deleted. Upload a different PDF to continue.")

    if st.session_state.get("document_indexed") is True:
    
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
                    with st.spinner("Searching PDF and generating answer..."):
                        answer = ask_backend(
                            question,
                            st.session_state["document_id"]
                        )

                    if answer["supported"]:
                        st.markdown(answer["answer"])

                        if answer["source_pages"]:
                            st.caption(
                                "Sources: PDF pages "
                                + ", ".join(map(str, answer["source_pages"]))
                            )
                    else:
                        st.warning(answer["answer"])

                    message_content = answer["answer"]

                    if answer["supported"] and answer["source_pages"]:
                        message_content += (
                            "\n\n**Sources:** PDF pages "
                            + ", ".join(map(str, answer["source_pages"]))
                        )

                except requests.exceptions.RequestException as error:
                    message_content = (
                        f"Sorry, the backend could not process your question: {error}"
                    )
                    st.error(message_content)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": message_content
            })