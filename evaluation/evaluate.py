import csv
import hashlib
import json
import random
import re
import sys
import time

from datetime import datetime
from pathlib import Path

import streamlit as st
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse existing RAG implementation
import app as rag


TEST_CASES = ROOT / "evaluation" / "test_cases.json"
REPORT_DIR = ROOT / "evaluation" / "reports"

# Gemini free-tier protection
MIN_REQUEST_INTERVAL_SECONDS = 6
MAX_RETRIES = 5
MAX_BACKOFF_SECONDS = 120

last_request_time = None


# ---------------------------------------------------------
# 2. GEMINI RATE LIMITING
# ---------------------------------------------------------

def wait_for_rate_limit():
    """
    Ensure sufficient time between Gemini API requests.

    This applies to:
    - RAG answer generation
    - Gemini correctness evaluation
    - Automatic retry attempts
    """

    global last_request_time

    now = time.monotonic()

    if last_request_time is not None:
        elapsed = now - last_request_time
        remaining = MIN_REQUEST_INTERVAL_SECONDS - elapsed

        if remaining > 0:
            time.sleep(remaining)

    last_request_time = time.monotonic()


# ---------------------------------------------------------
# 3. AUTOMATIC RETRY WITH EXPONENTIAL BACKOFF
# ---------------------------------------------------------

def extract_retry_delay(error_message):
    """
    Extract the API's suggested retry delay.

    Example:
    Please retry in 54.145 seconds
    """

    patterns = [
        r"retry in ([\d.]+)s",
        r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s?",
        r"retry_delay.*?seconds['\"]?\s*:\s*([\d.]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            error_message,
            flags=re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    return None


def call_gemini_with_retry(operation):
    """
    Execute a Gemini request safely.

    Retries temporary:
    - 429 RESOURCE_EXHAUSTED
    - 503 UNAVAILABLE

    Permanent errors are raised immediately.
    """

    for attempt in range(MAX_RETRIES + 1):

        wait_for_rate_limit()

        try:
            return operation()

        except Exception as exc:

            error_message = str(exc)

            retryable = any(
                keyword in error_message
                for keyword in [
                    "429",
                    "503",
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE"
                ]
            )

            if not retryable:
                raise

            if attempt >= MAX_RETRIES:
                raise

            server_delay = extract_retry_delay(
                error_message
            )

            exponential_delay = min(
                MAX_BACKOFF_SECONDS,
                5 * (2 ** attempt)
            )

            jitter = random.uniform(0, 2)

            if server_delay is not None:
                delay = max(
                    exponential_delay,
                    server_delay + 1
                )
            else:
                delay = exponential_delay + jitter

            st.warning(
                f"Gemini temporarily unavailable. "
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"in {delay:.1f} seconds."
            )

            time.sleep(delay)


# ---------------------------------------------------------
# 4. CHECKPOINT MANAGEMENT
# ---------------------------------------------------------

def save_checkpoint(checkpoint_path, checkpoint):
    """
    Save checkpoint with Windows-friendly retries.
    Uses a unique temporary file to avoid collisions.
    """

    import os
    import tempfile

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = json.dumps(
        checkpoint,
        indent=2,
        ensure_ascii=False
    )

    fd, temp_name = tempfile.mkstemp(
        prefix="checkpoint_",
        suffix=".tmp",
        dir=checkpoint_path.parent
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(data)

        for attempt in range(5):
            try:
                os.replace(temp_name, checkpoint_path)
                return

            except PermissionError:
                if attempt == 4:
                    raise

                time.sleep(0.5 * (attempt + 1))

    finally:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except PermissionError:
                pass


def load_checkpoint(checkpoint_path):

    if checkpoint_path.exists():

        return json.loads(
            checkpoint_path.read_text(
                encoding="utf-8"
            )
        )

    return {}


def create_checkpoint_path(
    pdf_id,
    selected_cases,
    use_judge
):
    """
    Different PDFs, datasets and judge settings
    use separate checkpoints.
    """

    configuration = {
        "pdf_id": pdf_id,
        "cases": selected_cases,
        "use_judge": use_judge,
        "answer_model": rag.ANSWER_MODEL,
        "evaluation_version": 4
    }

    config_hash = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:12]

    return (
        REPORT_DIR
        / f"checkpoint_{pdf_id[:12]}_{config_hash}.json"
    )


# ---------------------------------------------------------
# 5. ANSWER CORRECTNESS JUDGE
# ---------------------------------------------------------

class CorrectnessJudgment(BaseModel):

    correct: bool = Field(
        description=(
            "Whether the answer covers all essential "
            "expected facts without contradictions"
        )
    )

    reason: str = Field(
        description="Brief explanation of the judgment"
    )


def judge_correctness(
    question,
    expected_answer,
    actual_answer
):

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Compare the actual answer against the expected answer.

Rules:
1. Mark correct=true only when the essential expected
   facts are covered.
2. Paraphrasing is acceptable.
3. Do not require exact wording.
4. Material contradictions make the answer incorrect.
5. Do not use outside knowledge to fill missing facts.
6. Return a brief reason.

Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}
"""

    response = call_gemini_with_retry(
        lambda: rag.client.models.generate_content(
            model=rag.ANSWER_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CorrectnessJudgment,
                "temperature": 0
            }
        )
    )

    if not response.text:
        raise ValueError(
            "Correctness judge returned an empty response."
        )

    return CorrectnessJudgment.model_validate_json(
        response.text
    )


# ---------------------------------------------------------
# 6. CREATE EMPTY EVALUATION ROW
# ---------------------------------------------------------

def create_empty_row(case):

    return {
        "id": case["id"],
        "question": case["question"],
        "verified": case.get("verified", False),
        "expected_supported": case["supported"],
        "expected_pages": ",".join(
            map(str, case.get("relevant_pages", []))
        ),
        "retrieved_pages": "",
        "retrieval_hit": None,
        "predicted_supported": None,
        "support_correct": None,
        "answer_correct": None,
        "judge_reason": "",
        "answer": "",
        "retrieval_done": False,
        "generation_done": False,
        "judge_done": False,
        "error": ""
    }


# ---------------------------------------------------------
# 7. METRIC HELPERS
# ---------------------------------------------------------

def safe_rate(numerator, denominator):

    if denominator == 0:
        return None

    return round(
        numerator / denominator * 100,
        2
    )


def calculate_metrics(rows, use_judge):

    retrieval_cases = [
        row for row in rows
        if row["expected_supported"]
        and isinstance(row["retrieval_hit"], bool)
    ]

    classified_cases = [
        row for row in rows
        if isinstance(row["support_correct"], bool)
    ]

    unsupported_cases = [
        row for row in classified_cases
        if not row["expected_supported"]
    ]

    judged_cases = [
        row for row in rows
        if row["expected_supported"]
        and isinstance(row["answer_correct"], bool)
    ]

    errors = [
        row for row in rows
        if row["error"]
    ]

    metrics = {
        "Total questions": len(rows),

        "Retrieval Hit Rate (%)": safe_rate(
            sum(
                row["retrieval_hit"]
                for row in retrieval_cases
            ),
            len(retrieval_cases)
        ),

        "Support Classification Accuracy (%)": safe_rate(
            sum(
                row["support_correct"]
                for row in classified_cases
            ),
            len(classified_cases)
        ),

        "Unsupported Rejection Rate (%)": safe_rate(
            sum(
                row["predicted_supported"] is False
                for row in unsupported_cases
            ),
            len(unsupported_cases)
        ),

        "Answer Correctness (%)": (
            safe_rate(
                sum(
                    row["answer_correct"]
                    for row in judged_cases
                ),
                len(judged_cases)
            )
            if use_judge
            else None
        ),

        "Answers judged": len(judged_cases),

        "Errors": len(errors)
    }

    return metrics


# ---------------------------------------------------------
# 8. MAIN EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate(
    pdf_bytes,
    verified_only,
    use_judge
):

    cases = json.loads(
        TEST_CASES.read_text(
            encoding="utf-8"
        )
    )

    pdf_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # Check PDF exists in ChromaDB
    existing = rag.collection.get(
        where={"pdf_id": pdf_id},
        limit=1
    )

    if not existing["ids"]:
        raise ValueError(
            "PDF not indexed in ChromaDB. "
            "Upload it using the main RAG application first."
        )

    selected_cases = [
        case for case in cases
        if not verified_only
        or case.get("verified") is True
    ]

    selected_cases = [
        case for case in selected_cases
        if case["id"] in ["Q01", "Q10"]
    ]

    if not selected_cases:
        raise ValueError(
            "No evaluation questions selected."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = create_checkpoint_path(
        pdf_id,
        selected_cases,
        use_judge
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    st.info(
        f"Checkpoint: {checkpoint_path.name}"
    )

    progress = st.progress(0)
    status = st.empty()

    rows = []

    for index, case in enumerate(
        selected_cases,
        start=1
    ):

        case_id = case["id"]

        status.write(
            f"Evaluating {index}/{len(selected_cases)}: "
            f"{case_id}"
        )

        # Resume previously saved progress
        row = checkpoint.get(
            case_id,
            create_empty_row(case)
        )

        row["error"] = ""

        try:

            # -----------------------------------------
            # STAGE A: RETRIEVAL
            # -----------------------------------------

            if not row["retrieval_done"]:

                top_chunks, relevant_chunks = (
                    rag.retrieve_chunks(
                        case["question"],
                        pdf_id
                    )
                )

                if case["id"] in ["Q01", "Q10"]:
                    st.write(f"DEBUG {case['id']} retrieved evidence:")
                    for chunk in relevant_chunks:
                        st.write({
                            "page": chunk["page"],
                            "method": chunk["retrieval_method"],
                            "text": chunk["text"]
                        })

                retrieved_pages = sorted({
                    chunk["page"]
                    for chunk in top_chunks
                })

                row["retrieved_pages"] = ",".join(
                    map(str, retrieved_pages)
                )

                if (
                    case["supported"]
                    and case.get("relevant_pages")
                ):

                    row["retrieval_hit"] = bool(
                        set(case["relevant_pages"])
                        & set(retrieved_pages)
                    )

                row["retrieval_done"] = True

                checkpoint[case_id] = row.copy()

                save_checkpoint(
                    checkpoint_path,
                    checkpoint
                )

            # Retrieval is local, so reconstruct
            # evidence when resuming generation.
            relevant_chunks = None

            # -----------------------------------------
            # STAGE B: ANSWER GENERATION
            # -----------------------------------------

            if not row["generation_done"]:

                _, relevant_chunks = (
                    rag.retrieve_chunks(
                        case["question"],
                        pdf_id
                    )
                )

                if relevant_chunks:

                    result = call_gemini_with_retry(
                        lambda: rag.generate_answer(
                            case["question"],
                            relevant_chunks
                        )
                    )

                    row["predicted_supported"] = (
                        result.supported
                    )

                    row["answer"] = result.answer

                else:

                    row["predicted_supported"] = False

                    row["answer"] = (
                        "Insufficient evidence in the PDF."
                    )

                row["support_correct"] = (
                    row["predicted_supported"]
                    == case["supported"]
                )

                row["generation_done"] = True

                checkpoint[case_id] = row.copy()

                save_checkpoint(
                    checkpoint_path,
                    checkpoint
                )

            # -----------------------------------------
            # STAGE C: ANSWER CORRECTNESS
            # -----------------------------------------

            if not row["judge_done"]:

                if (
                    use_judge
                    and case["supported"]
                    and row["predicted_supported"]
                ):

                    judgment = judge_correctness(
                        case["question"],
                        case["expected_answer"],
                        row["answer"]
                    )

                    row["answer_correct"] = (
                        judgment.correct
                    )

                    row["judge_reason"] = (
                        judgment.reason
                    )

                elif (
                    case["supported"]
                    and not row["predicted_supported"]
                ):

                    if use_judge:
                        row["answer_correct"] = False

                    row["judge_reason"] = (
                        "Supported question was rejected."
                    )

                row["judge_done"] = True

                checkpoint[case_id] = row.copy()

                save_checkpoint(
                    checkpoint_path,
                    checkpoint
                )

        except Exception as exc:

            row["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            st.error(
                f"{case_id} failed: {row['error']}"
            )

        # Save progress after every question
        checkpoint[case_id] = row.copy()

        save_checkpoint(
            checkpoint_path,
            checkpoint
        )

        rows.append(row)

        progress.progress(
            index / len(selected_cases)
        )

    # -------------------------------------------------
    # 9. GENERATE CSV REPORT
    # -------------------------------------------------

    report_path = (
        REPORT_DIR
        / f"rag_eval_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------
    # 10. DISPLAY METRICS
    # -------------------------------------------------

    metrics = calculate_metrics(
        rows,
        use_judge
    )

    status.success(
        "Evaluation finished."
    )

    st.subheader("Evaluation Metrics")

    st.json(metrics)

    st.subheader("Detailed Results")

    st.dataframe(
        rows,
        use_container_width=True
    )

    st.success(
        f"CSV report saved to: {report_path}"
    )

    st.download_button(
        "Download Evaluation Report",
        data=report_path.read_bytes(),
        file_name=report_path.name,
        mime="text/csv"
    )

    st.caption(
        "Metrics use completed stages only. "
        "Page-level retrieval hits depend on the "
        "quality of the reference-page annotations."
    )


# ---------------------------------------------------------
# 11. STREAMLIT INTERFACE
# ---------------------------------------------------------

st.title("RAG Evaluation")

st.write(
    "Evaluate your existing ChromaDB + Gemini "
    "single-PDF RAG application."
)

st.caption(
    "Includes API rate limiting, automatic retries, "
    "and checkpoint-based resume."
)

uploaded_file = st.file_uploader(
    "Upload the same PDF already indexed in ChromaDB",
    type=["pdf"]
)

verified_only = st.checkbox(
    "Use only verified test cases",
    value=True
)

use_judge = st.checkbox(
    "Use Gemini to judge answer correctness",
    value=True
)

if "evaluation_running" not in st.session_state:
    st.session_state.evaluation_running = False

if st.button(
    "Run / Resume Evaluation",
    disabled=(
        uploaded_file is None
        or st.session_state.evaluation_running
    )
):

    st.session_state.evaluation_running = True

    try:
        evaluate(
            pdf_bytes=uploaded_file.getvalue(),
            verified_only=verified_only,
            use_judge=use_judge
        )

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

    finally:
        st.session_state.evaluation_running = False