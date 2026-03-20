import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"
ANSWER_DIR = PROJECT_ROOT / "answer"


def ensure_dirs() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file() -> None:
    for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def get_openai_api_key() -> str:
    load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Add it to .env or your shell environment.")
    return api_key


def chat_completion(
    messages: Sequence[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> str:
    api_key = get_openai_api_key()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data, path: Path) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_task1_data(
    *,
    chapter_number: int,
) -> Tuple[List[Dict[str, str]], str, List[Dict[str, str]]]:
    chunks_path = ARTEFACTS_DIR / f"chapter_{chapter_number}_chunks.json"
    clean_text_path = ARTEFACTS_DIR / f"chapter_{chapter_number}_cleaned.txt"
    qa_path = ARTEFACTS_DIR / f"qa_pairs_chapter_{chapter_number}.json"

    chunks = load_json(chunks_path)
    document_text = clean_text_path.read_text(encoding="utf-8")
    qa_pairs = load_json(qa_path)
    return chunks, document_text, qa_pairs


def chunk_to_text(chunk: Dict[str, str]) -> str:
    return chunk.get("text", "")


def build_contextualized_chunks(
    *,
    chunks: Sequence[Dict[str, str]],
    document_text: str,
    title: str,
    model: str = "gpt-4o-mini",
    output_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    contextualized = []
    document_excerpt = document_text[:4000]

    for chunk in chunks:
        prompt = (
            f"Title: {title}\n\n"
            f"Document excerpt:\n{document_excerpt}\n\n"
            f"Chunk:\n{chunk_to_text(chunk)}\n\n"
            "Provide a brief 1-2 sentence context summary describing what this chunk is about "
            'relative to the whole chapter. Start with: "This chunk discusses ..."'
        )
        context = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=120,
        )
        contextualized.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk_to_text(chunk),
                "context": context,
                "retrieval_text": f"{context}\n\n{chunk_to_text(chunk)}",
            }
        )

    if output_path is not None:
        save_json(contextualized, output_path)
    return contextualized


class TfidfRetriever:
    def __init__(self, documents: Sequence[str]) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[int, float]]:
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        ranked = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


def retrieve_top_chunks(
    *,
    question: str,
    chunks: Sequence[Dict[str, str]],
    top_k: int = 4,
    use_contextual_text: bool = False,
) -> List[Dict[str, str]]:
    documents = [
        chunk["retrieval_text"] if use_contextual_text else chunk_to_text(chunk)
        for chunk in chunks
    ]
    retriever = TfidfRetriever(documents)
    ranked = retriever.search(question, top_k=top_k)

    selected = []
    for idx, score in ranked:
        enriched = dict(chunks[idx])
        enriched["retrieval_score"] = float(score)
        selected.append(enriched)
    return selected


def build_answer_prompt(question: str, retrieved_chunks: Sequence[Dict[str, str]]) -> str:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Chunk {chunk['chunk_id']}]\n{chunk_to_text(chunk)}"
        )

    joined_context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the retrieved context.\n"
        "Keep the answer concise and factual.\n"
        "Do not mention any information not supported by the retrieved text.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{joined_context}"
    )


def answer_question(
    *,
    question: str,
    retrieved_chunks: Sequence[Dict[str, str]],
    model: str = "gpt-4o-mini",
) -> str:
    prompt = build_answer_prompt(question, retrieved_chunks)
    return chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=220,
    )


def run_rag_method(
    *,
    qa_pairs: Sequence[Dict[str, str]],
    chunks: Sequence[Dict[str, str]],
    model: str = "gpt-4o-mini",
    top_k: int = 4,
    use_contextual_text: bool = False,
    answer_key: str = "naive_rag_answer",
) -> List[Dict[str, str]]:
    rows = []
    for item in qa_pairs:
        question = item["question"]
        ground_truth_answer = item["ground_truth_answer"]
        retrieved_chunks = retrieve_top_chunks(
            question=question,
            chunks=chunks,
            top_k=top_k,
            use_contextual_text=use_contextual_text,
        )
        answer = answer_question(
            question=question,
            retrieved_chunks=retrieved_chunks,
            model=model,
        )
        rows.append(
            {
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                answer_key: answer,
                "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            }
        )
    return rows


def strip_citations(text: str) -> str:
    return re.sub(r"\[cite:[^\]]+\]", "", text).strip()


def tokenize(text: str) -> List[str]:
    text = strip_citations(text.lower())
    return re.findall(r"[a-z0-9]+", text)


def rouge_n(prediction: str, reference: str, n: int) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    def build_ngrams(tokens: Sequence[str]) -> Dict[Tuple[str, ...], int]:
        counts: Dict[Tuple[str, ...], int] = {}
        for idx in range(len(tokens) - n + 1):
            gram = tuple(tokens[idx : idx + n])
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    pred_counts = build_ngrams(pred_tokens)
    ref_counts = build_ngrams(ref_tokens)
    overlap = sum(min(pred_counts[gram], ref_counts.get(gram, 0)) for gram in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    return (2 * precision * recall) / (precision + recall)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_results(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    naive_r1, naive_r2, naive_rl = [], [], []
    contextual_r1, contextual_r2, contextual_rl = [], [], []

    for row in rows:
        reference = row["ground_truth_answer"]
        naive_answer = row["naive_rag_answer"]
        contextual_answer = row["contextual_retrieval_answer"]

        naive_r1.append(rouge_n(naive_answer, reference, 1))
        naive_r2.append(rouge_n(naive_answer, reference, 2))
        naive_rl.append(rouge_l(naive_answer, reference))

        contextual_r1.append(rouge_n(contextual_answer, reference, 1))
        contextual_r2.append(rouge_n(contextual_answer, reference, 2))
        contextual_rl.append(rouge_l(contextual_answer, reference))

    return {
        "naive_rag": {
            "rouge_1": average(naive_r1),
            "rouge_2": average(naive_r2),
            "rouge_l": average(naive_rl),
        },
        "contextual_retrieval": {
            "rouge_1": average(contextual_r1),
            "rouge_2": average(contextual_r2),
            "rouge_l": average(contextual_rl),
        },
    }


def merge_method_results(
    naive_rows: Sequence[Dict[str, str]],
    contextual_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged = []
    for naive_row, contextual_row in zip(naive_rows, contextual_rows):
        merged.append(
            {
                "question": naive_row["question"],
                "ground_truth_answer": naive_row["ground_truth_answer"],
                "naive_rag_answer": naive_row["naive_rag_answer"],
                "contextual_retrieval_answer": contextual_row["contextual_retrieval_answer"],
                "naive_retrieved_chunk_ids": naive_row["retrieved_chunk_ids"],
                "contextual_retrieved_chunk_ids": contextual_row["retrieved_chunk_ids"],
            }
        )
    return merged
