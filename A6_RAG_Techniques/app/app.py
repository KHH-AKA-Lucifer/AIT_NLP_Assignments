from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_utils import answer_question, load_json, retrieve_top_chunks


ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"

CHAPTER_NUMBER = 7
MODEL_NAME = "gpt-4o-mini"
TOP_K = 4

CHUNKS_PATH = ARTEFACTS_DIR / f"chapter_{CHAPTER_NUMBER}_chunks.json"
CONTEXTUAL_CHUNKS_PATH = ARTEFACTS_DIR / f"chapter_{CHAPTER_NUMBER}_contextual_chunks.json"


app = Flask(__name__, template_folder=str(APP_DIR / "templates"))


def load_chunk_store() -> tuple[list[dict], bool]:
    """Prefer contextual chunks when they already exist."""
    if CONTEXTUAL_CHUNKS_PATH.exists():
        return load_json(CONTEXTUAL_CHUNKS_PATH), True
    if CHUNKS_PATH.exists():
        return load_json(CHUNKS_PATH), False
    raise FileNotFoundError(
        "No chunk cache was found. Run Task 1 and Task 2 first so the artefacts exist."
    )


def get_chunk_store() -> tuple[list[dict], bool]:
    chunks = app.config.get("chunk_store")
    use_contextual = app.config.get("use_contextual")
    if chunks is None or use_contextual is None:
        chunks, use_contextual = load_chunk_store()
        app.config["chunk_store"] = chunks
        app.config["use_contextual"] = use_contextual
    return chunks, use_contextual


def build_source_cards(retrieved_chunks: list[dict]) -> list[dict]:
    cards = []
    for chunk in retrieved_chunks:
        preview = " ".join(chunk.get("text", "").split())
        if len(preview) > 260:
            preview = preview[:260].rstrip() + "..."

        cards.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": round(float(chunk.get("retrieval_score", 0.0)), 4),
                "context": chunk.get("context", ""),
                "preview": preview,
            }
        )
    return cards


@app.get("/")
def index():
    _, use_contextual = get_chunk_store()
    return render_template(
        "index.html",
        chapter_number=CHAPTER_NUMBER,
        model_name=MODEL_NAME,
        retrieval_mode="Contextual Retrieval" if use_contextual else "Naive RAG",
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Please enter a question before sending."}), 400

    try:
        chunks, use_contextual = get_chunk_store()
        retrieved_chunks = retrieve_top_chunks(
            question=question,
            chunks=chunks,
            top_k=TOP_K,
            use_contextual_text=use_contextual,
        )
        answer = answer_question(
            question=question,
            retrieved_chunks=retrieved_chunks,
            model=MODEL_NAME,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "question": question,
            "answer": answer,
            "mode": "Contextual Retrieval" if use_contextual else "Naive RAG",
            "model": MODEL_NAME,
            "sources": build_source_cards(retrieved_chunks),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
