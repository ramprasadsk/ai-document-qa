import os
import hashlib
import requests
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
)

BACKEND_API_KEY = st.secrets.get(
    "BACKEND_API_KEY",
    os.getenv("BACKEND_API_KEY", "")
)

API_HEADERS = {
    "X-API-Key": BACKEND_API_KEY
}

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Document Q&A")
st.write("Upload one PDF and ask questions based on its content.")


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
        headers=API_HEADERS,
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
        headers=API_HEADERS,
        timeout=120
    )

    response.raise_for_status()

    return response.json()

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
        st.session_state["document_indexed"] = None
        st.session_state["messages"] = []

    
    if st.session_state.get("document_indexed") is None:
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

            st.session_state["document_indexed"] = True

        except requests.exceptions.RequestException as error:
            st.error(f"FastAPI backend error: {error}")
            st.stop()

    
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
                    answer = ask_backend(question, pdf_id)

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
