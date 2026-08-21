"""Run a reproducible baseline evaluation over ``evaluation/questions.json``.

This is intentionally a small measurement harness, not a full evaluation
framework. It reports retrieval hit rate, latency, lexical answer overlap,
embedding similarity, and answer/context token overlap. The lexical metrics
are proxies and should not be interpreted as human correctness judgments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.services.embeddings import embed_texts
from app.services.llm import LLMService
from app.services.rag_pipeline import RAGPipeline

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
REFUSAL_PHRASES = (
    "not enough",
    "could not find enough",
    "couldn't find enough",
    "insufficient",
    "cannot answer",
    "can't answer",
    "could not answer",
    "couldn't answer",
    "not provided",
    "not contain",
    "no relevant information",
    "unsupported",
    "do not know",
)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def token_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = _tokens(reference)
    hypothesis_tokens = _tokens(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    reference_counts = {token: reference_tokens.count(token) for token in set(reference_tokens)}
    hypothesis_counts = {token: hypothesis_tokens.count(token) for token in set(hypothesis_tokens)}
    overlap = sum(min(reference_counts.get(token, 0), count) for token, count in hypothesis_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def embedding_cosine(reference: str, hypothesis: str) -> float | None:
    if not reference.strip() or not hypothesis.strip():
        return None
    try:
        vectors = embed_texts([reference, hypothesis])
    except Exception:
        return None
    first, second = vectors[0], vectors[1]
    denominator = float((first @ first) ** 0.5 * (second @ second) ** 0.5)
    if denominator == 0.0:
        return None
    return float(first @ second / denominator)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _latency_summary(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(record["timing"][key]) for record in records if record.get("timing")]
    return {
        "mean_ms": mean(values) if values else None,
        "median_ms": median(values) if values else None,
        "p95_ms": _percentile(values, 0.95),
    }


def evaluate_question(pipeline: RAGPipeline, item: dict[str, Any]) -> dict[str, Any]:
    question = str(item["question"])
    expected_source = item.get("expected_source")
    expected_pages = {int(page) for page in item.get("expected_pages", [])}
    supported = bool(item.get("supported", True))

    try:
        response = pipeline.execute(question)
    except Exception as exc:  # pragma: no cover - provider/network dependent
        return {
            "id": item.get("id"),
            "question": question,
            "supported": supported,
            "error": str(exc),
            "retrieval_hit": False,
        }

    retrieved = response.results
    source_hit = any(result.get("filename") == expected_source for result in retrieved) if expected_source else False
    page_hit = any(
        result.get("filename") == expected_source and result.get("page_number") in expected_pages
        for result in retrieved
    ) if expected_source and expected_pages else None
    context = "\n".join(str(result.get("text", "")) for result in retrieved)
    answer = response.answer
    answer_tokens = set(_tokens(answer))
    context_tokens = set(_tokens(context))
    groundedness = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    refusal_detected = any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)

    if supported:
        answer_token_f1 = token_f1(str(item.get("expected_answer", "")), answer)
        answer_cosine = embedding_cosine(str(item.get("expected_answer", "")), answer)
        answer_correct = answer_token_f1 >= 0.35
    else:
        answer_token_f1 = None
        answer_cosine = None
        answer_correct = refusal_detected

    return {
        "id": item.get("id"),
        "question": question,
        "supported": supported,
        "expected_source": expected_source,
        "expected_pages": sorted(expected_pages),
        "answer": answer,
        "retrieved": [
            {
                "document_id": result.get("document_id"),
                "filename": result.get("filename"),
                "page_number": result.get("page_number"),
                "rank": result.get("rank"),
                "score": result.get("score"),
                "chunk_id": result.get("chunk_id"),
            }
            for result in retrieved
        ],
        "retrieval_hit": source_hit,
        "page_hit": page_hit,
        "refusal_detected": refusal_detected,
        "answer_correct": answer_correct,
        "answer_token_f1": answer_token_f1,
        "answer_embedding_cosine": answer_cosine,
        "groundedness_token_overlap": groundedness,
        "timing": response.timing.to_dict(),
        "provider": response.provider,
        "model": response.model,
    }


def run(
    questions_path: Path,
    output_path: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    effective_provider = provider or settings.llm_provider
    effective_model = model or settings.llm_model
    llm_service = LLMService(provider=effective_provider, model=effective_model, token=settings.hf_token)
    pipeline = RAGPipeline(llm_service=llm_service)
    records = [evaluate_question(pipeline, item) for item in questions]
    completed = [record for record in records if "timing" in record]
    supported_records = [record for record in completed if record.get("supported")]
    unsupported_records = [record for record in completed if not record.get("supported")]
    f1_values = [record["answer_token_f1"] for record in supported_records if record.get("answer_token_f1") is not None]
    cosine_values = [record["answer_embedding_cosine"] for record in supported_records if record.get("answer_embedding_cosine") is not None]
    groundedness_values = [float(record["groundedness_token_overlap"]) for record in completed]

    result: dict[str, Any] = {
        "experiment": "baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(questions_path.resolve()),
        "configuration": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "embedding_model": settings.embedding_model,
            "llm_provider": effective_provider,
            "llm_model": effective_model,
            "similarity_threshold": settings.similarity_threshold,
        },
        "metric_notes": {
            "retrieval_relevance": "retrieval_hit is a filename hit@top_k; page_hit requires a returned chunk page to overlap expected_pages.",
            "answer_token_f1": "lexical overlap proxy, not a human correctness judgment.",
            "answer_embedding_cosine": "cosine similarity using the configured embedding model; diagnostic only.",
            "groundedness_token_overlap": "fraction of answer tokens also present in retrieved context; diagnostic only.",
            "unsupported_answer_correct": "true only when the answer contains a refusal/insufficiency phrase.",
        },
        "summary": {
            "question_count": len(questions),
            "completed_count": len(completed),
            "failure_count": len(records) - len(completed),
            "retrieval_hit_at_k": mean(bool(record.get("retrieval_hit")) for record in supported_records) if supported_records else None,
            "page_hit_at_k": mean(bool(record.get("page_hit")) for record in supported_records if record.get("page_hit") is not None) if supported_records else None,
            "supported_answer_token_f1_mean": mean(f1_values) if f1_values else None,
            "supported_answer_embedding_cosine_mean": mean(cosine_values) if cosine_values else None,
            "groundedness_token_overlap_mean": mean(groundedness_values) if groundedness_values else None,
            "unsupported_refusal_rate": mean(bool(record.get("answer_correct")) for record in unsupported_records) if unsupported_records else None,
            "retrieval_latency": _latency_summary(completed, "retrieval_ms"),
            "generation_latency": _latency_summary(completed, "generation_ms"),
            "total_latency": _latency_summary(completed, "total_ms"),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "baseline.json")
    parser.add_argument("--provider", help="Override the configured provider, e.g. mock when HF credentials are unavailable.")
    parser.add_argument("--model", help="Override the configured model name.")
    args = parser.parse_args()
    result = run(args.questions, args.output, provider=args.provider, model=args.model)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
