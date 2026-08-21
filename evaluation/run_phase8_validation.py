"""Run Phase 8 Hugging Face validation without fabricating live results.

The script uses the configured Hugging Face provider only when ``HF_TOKEN`` is
present in ``backend/.env`` (or the process environment). With no token it
writes explicit blocked artifacts and performs only local configuration/index
checks. Phase 7 result files are never modified.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
REAL_RESULT_PATH = RESULTS_DIR / "real_llm_baseline.json"
CITATION_RESULT_PATH = RESULTS_DIR / "source_citation_correctness.json"
FAILURE_RESULT_PATH = RESULTS_DIR / "failure_analysis.json"
COMPARISON_RESULT_PATH = RESULTS_DIR / "real_vs_mock_comparison.json"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.llm import LLMConfigurationError, LLMProviderError
from evaluation.run_baseline import run as run_baseline


CLASSIFICATION_MAP = {
    "direct_factual": "supported_factual",
    "conceptual": "conceptual",
    "comparison": "comparison",
    "multi_sentence": "multi_sentence",
    "unsupported": "unsupported",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _configuration() -> dict[str, Any]:
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "embedding_model": settings.embedding_model,
        "similarity_threshold": settings.similarity_threshold,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_temperature": settings.llm_temperature,
    }


def _index_stats() -> dict[str, Any]:
    index_path = Path(settings.faiss_index_path)
    metadata_path = index_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    unique_keys: set[str] = set()
    duplicate_keys = 0
    for item in metadata:
        key = f"{item.get('content_hash', '')}|{item.get('chunk_id', '')}"
        if key in unique_keys:
            duplicate_keys += 1
        unique_keys.add(key)
    pages = [item.get("page_number") for item in metadata if item.get("page_number") is not None]
    return {
        "index_path": str(index_path),
        "faiss_index_exists": (index_path / "faiss_index.bin").exists(),
        "metadata_exists": metadata_path.exists(),
        "document_count": len({item.get("document_id") for item in metadata}),
        "chunk_count": len(metadata),
        "unique_document_ids": sorted({str(item.get("document_id")) for item in metadata}),
        "unique_content_hashes": len({item.get("content_hash") for item in metadata if item.get("content_hash")}),
        "duplicate_content_hash_chunk_keys": duplicate_keys,
        "missing_page_numbers": sum(item.get("page_number") is None for item in metadata),
        "min_page_number": min(pages) if pages else None,
        "max_page_number": max(pages) if pages else None,
    }


def _sanitize_error(error: Exception | str) -> str:
    message = str(error)
    if settings.hf_token:
        message = message.replace(settings.hf_token, "[REDACTED]")
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    return message


def _questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def _question_metadata() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in _questions()}


def _smoke_record(response: Any) -> dict[str, Any]:
    return {
        "question": response.question,
        "answer": response.answer,
        "provider": response.provider,
        "model": response.model,
        "sources": [source.to_dict() for source in response.sources],
        "retrieved_pages": [source.page_number for source in response.sources],
        "timing": response.timing.to_dict(),
    }


def _run_smoke_test() -> dict[str, Any]:
    from app.services.rag_pipeline import RAGPipeline

    questions = _questions()
    supported = next(item for item in questions if item.get("id") == "cloud-005")
    unsupported = next(item for item in questions if not item.get("supported"))
    try:
        pipeline = RAGPipeline()
        supported_response = pipeline.execute(str(supported["question"]))
        unsupported_response = pipeline.execute(str(unsupported["question"]))
        return {
            "status": "passed",
            "supported": _smoke_record(supported_response),
            "unsupported": _smoke_record(unsupported_response),
            "unsupported_refused": not bool(unsupported_response.results),
            "provider_verified": supported_response.provider == "huggingface",
            "model_verified": supported_response.model == settings.llm_model,
        }
    except (LLMConfigurationError, LLMProviderError, RuntimeError, ValueError) as exc:
        return {"status": "failed", "error": _sanitize_error(exc)}
    except Exception as exc:  # pragma: no cover - defensive provider boundary
        return {"status": "failed", "error": _sanitize_error(exc)}


def _augment_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = _question_metadata()
    augmented: list[dict[str, Any]] = []
    for record in records:
        item = questions.get(str(record.get("id")), {})
        retrieved = record.get("retrieved", [])
        record = dict(record)
        record.update(
            {
                "reference_answer": item.get("expected_answer"),
                "question_type": item.get("question_type"),
                "classification": CLASSIFICATION_MAP.get(
                    str(item.get("question_type")),
                    "unsupported" if not item.get("supported", True) else "supported_factual",
                ),
                "retrieved_pages": [entry.get("page_number") for entry in retrieved],
                "retrieval_scores": [entry.get("score") for entry in retrieved],
            }
        )
        augmented.append(record)
    return augmented


def _citation_summary(records: list[dict[str, Any]], *, status: str) -> dict[str, Any]:
    supported = [record for record in records if record.get("supported") and "timing" in record]
    unsupported = [record for record in records if not record.get("supported") and "timing" in record]
    document_hits = [bool(record.get("retrieval_hit")) for record in supported]
    page_hits = [bool(record.get("page_hit")) for record in supported if record.get("page_hit") is not None]
    groundedness = [float(record.get("groundedness_token_overlap", 0.0)) for record in supported]
    low_groundedness = [record for record in supported if float(record.get("groundedness_token_overlap", 0.0)) < 0.5]
    return {
        "status": status,
        "supported_question_count": len(supported),
        "correct_document_rate": mean(document_hits) if document_hits else None,
        "correct_page_rate": mean(page_hits) if page_hits else None,
        "mean_supported_groundedness_token_overlap": mean(groundedness) if groundedness else None,
        "low_groundedness_count": len(low_groundedness),
        "unsupported_question_count": len(unsupported),
        "unsupported_refusal_rate": (
            mean(bool(record.get("refusal_detected")) for record in unsupported) if unsupported else None
        ),
        "unsupported_questions_with_context": sum(bool(record.get("retrieved")) for record in unsupported),
        "answer_agreement_note": (
            "Groundedness is a token-overlap diagnostic. It does not prove semantic entailment or absence of unsupported claims; human review or an entailment evaluator is required."
        ),
    }


def _failure_analysis(records: list[dict[str, Any]], *, status: str, reason: str | None = None) -> dict[str, Any]:
    if status != "completed":
        return {
            "experiment": "phase_8_failure_analysis",
            "status": "blocked",
            "analysis_basis": "No real-provider records were produced.",
            "reason": reason,
            "representative_examples": [],
            "non_fabrication_note": "Real failure causes are not inferred without real LLM outputs.",
        }

    examples: list[dict[str, Any]] = []
    for record in records:
        if "timing" not in record:
            examples.append(
                {
                    "question_id": record.get("id"),
                    "question": record.get("question"),
                    "failure_class": "provider_or_evaluation_error",
                    "evidence": record.get("error"),
                    "cause_confidence": "observed",
                }
            )
            continue
        if not record.get("supported") and not record.get("refusal_detected"):
            examples.append(
                {
                    "question_id": record.get("id"),
                    "question": record.get("question"),
                    "failure_class": "unsupported_question_handling",
                    "evidence": {"retrieved_count": len(record.get("retrieved", [])), "answer": record.get("answer")},
                    "cause_confidence": "observed",
                }
            )
        elif record.get("supported") and not record.get("retrieval_hit"):
            examples.append(
                {
                    "question_id": record.get("id"),
                    "question": record.get("question"),
                    "failure_class": "retrieval_failure",
                    "evidence": {"retrieved_count": len(record.get("retrieved", [])), "page_hit": record.get("page_hit")},
                    "cause_confidence": "observed",
                }
            )
        elif record.get("supported") and not record.get("page_hit"):
            examples.append(
                {
                    "question_id": record.get("id"),
                    "question": record.get("question"),
                    "failure_class": "page_metadata_or_retrieval_mismatch",
                    "evidence": {"retrieved_pages": record.get("retrieved_pages"), "expected_pages": record.get("expected_pages")},
                    "cause_confidence": "diagnostic",
                }
            )
        elif record.get("supported") and float(record.get("answer_token_f1") or 0.0) < 0.35:
            examples.append(
                {
                    "question_id": record.get("id"),
                    "question": record.get("question"),
                    "failure_class": "generation_or_prompt_quality",
                    "evidence": {
                        "answer_token_f1": record.get("answer_token_f1"),
                        "groundedness_token_overlap": record.get("groundedness_token_overlap"),
                    },
                    "cause_confidence": "diagnostic; retrieval was present",
                }
            )
        if len(examples) >= 8:
            break
    return {
        "experiment": "phase_8_failure_analysis",
        "status": "completed",
        "representative_examples": examples,
        "classification_notes": {
            "retrieval_failure": "No expected source was returned.",
            "page_metadata_or_retrieval_mismatch": "Source was returned but no expected page was present; chunks may span pages.",
            "generation_or_prompt_quality": "Retrieved source was present but lexical answer overlap was weak; this is not a causal proof.",
            "unsupported_question_handling": "Unsupported input did not produce a refusal.",
        },
    }


def _mock_comparison() -> dict[str, Any]:
    summary_path = RESULTS_DIR / "experiment_summary.json"
    phase7 = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    baseline = phase7.get("baseline", {})
    optimized = next(
        (
            item
            for item in phase7.get("experiments", {}).get("similarity_threshold", [])
            if item.get("configuration", {}).get("similarity_threshold") == 0.15
        ),
        {},
    )

    def metrics(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "retrieval_hit_at_k": source.get("retrieval_hit_at_k"),
            "page_hit_at_k": source.get("page_hit_at_k"),
            "supported_answer_token_f1_mean": source.get("supported_answer_token_f1_mean"),
            "supported_answer_embedding_cosine_mean": source.get("supported_answer_embedding_cosine_mean"),
            "unsupported_refusal_rate": source.get("unsupported_refusal_rate"),
            "mean_retrieval_latency_ms": (source.get("retrieval_latency") or {}).get("mean_ms"),
            "mean_total_latency_ms": (source.get("total_latency") or {}).get("mean_ms"),
        }

    return {
        "experiment": "phase_8_real_vs_mock_comparison",
        "status": "real_evaluation_blocked",
        "mock_provider": {
            "source": "evaluation/results/experiment_summary.json",
            "evaluation_role": "deterministic retrieval optimization; not representative of real LLM generation",
            "baseline_metrics": metrics(baseline),
            "optimized_metrics": metrics(optimized),
        },
        "real_huggingface": {
            "status": "not_available",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "metrics": None,
            "reason": "HF_TOKEN is not configured; no provider calls were made.",
        },
        "comparison_note": "No real-vs-mock quality or latency comparison is claimed because the real evaluation did not run.",
    }


def _blocked_result(reason: str) -> dict[str, Any]:
    questions = _questions()
    citation = _citation_summary([], status="blocked")
    citation["supported_question_count"] = sum(bool(item.get("supported")) for item in questions)
    citation["unsupported_question_count"] = sum(not bool(item.get("supported")) for item in questions)
    result = {
        "experiment": "phase_8_real_llm_validation",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "blocked",
        "live_validation": False,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "configuration": _configuration(),
        "dataset": str(QUESTIONS_PATH.resolve()),
        "question_count": len(questions),
        "blocked_reason": reason,
        "smoke_test": {"status": "not_run", "reason": reason},
        "non_live_validation": {
            "configuration_loaded": True,
            "provider_is_huggingface": settings.llm_provider == "huggingface",
            "index": _index_stats(),
            "questions_loaded_without_modification": True,
        },
        "metric_groups": {
            "retrieval": None,
            "generation_answer_quality": None,
        },
        "records": [],
        "source_citation_correctness": citation,
        "security": {
            "token_logged": False,
            "token_written_to_result": False,
            "env_file_ignored": True,
        },
    }
    return result


def _run_live_validation() -> dict[str, Any]:
    smoke = _run_smoke_test()
    if smoke.get("status") != "passed":
        return {
            **_blocked_result("Hugging Face smoke test failed; no full real evaluation was claimed."),
            "status": "failed",
            "live_validation": False,
            "smoke_test": smoke,
        }

    try:
        raw = run_baseline(
            QUESTIONS_PATH,
            REAL_RESULT_PATH,
            provider="huggingface",
            model=settings.llm_model,
        )
    except Exception as exc:  # pragma: no cover - live provider boundary
        failed = _blocked_result("Full real evaluation failed after the smoke test; no successful metrics were claimed.")
        failed.update(
            {
                "status": "failed",
                "live_validation": False,
                "smoke_test": smoke,
                "error": _sanitize_error(exc),
            }
        )
        return failed
    records = _augment_records(raw.get("records", []))
    completed = [record for record in records if "timing" in record]
    summary = raw.get("summary", {})
    result = dict(raw)
    result.update(
        {
            "experiment": "phase_8_real_llm_validation",
            "status": "completed" if len(completed) == len(records) else "completed_with_failures",
            "live_validation": True,
            "configuration": _configuration(),
            "smoke_test": smoke,
            "records": records,
            "metric_groups": {
                "retrieval": {
                    "filename_hit_at_k": summary.get("retrieval_hit_at_k"),
                    "page_hit_at_k": summary.get("page_hit_at_k"),
                    "mean_retrieval_latency_ms": (summary.get("retrieval_latency") or {}).get("mean_ms"),
                    "total_p95_latency_ms": (summary.get("total_latency") or {}).get("p95_ms"),
                },
                "generation_answer_quality": {
                    "supported_answer_token_f1_mean": summary.get("supported_answer_token_f1_mean"),
                    "supported_answer_embedding_cosine_mean": summary.get("supported_answer_embedding_cosine_mean"),
                    "groundedness_token_overlap_mean": summary.get("groundedness_token_overlap_mean"),
                    "unsupported_refusal_rate": summary.get("unsupported_refusal_rate"),
                    "mean_generation_latency_ms": (summary.get("generation_latency") or {}).get("mean_ms"),
                    "mean_total_latency_ms": (summary.get("total_latency") or {}).get("mean_ms"),
                },
            },
        }
    )
    result["source_citation_correctness"] = _citation_summary(records, status=result["status"])
    result["security"] = {
        "token_logged": False,
        "token_written_to_result": False,
        "env_file_ignored": True,
    }
    _write_json(REAL_RESULT_PATH, result)
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if settings.llm_provider != "huggingface":
        result = _blocked_result("LLM_PROVIDER is not set to huggingface; live validation was not attempted.")
    elif not settings.hf_token:
        result = _blocked_result("HF_TOKEN is not configured in backend/.env; no Hugging Face calls were made.")
    else:
        result = _run_live_validation()

    _write_json(REAL_RESULT_PATH, result)
    citation = result.get("source_citation_correctness", _citation_summary([], status="blocked"))
    _write_json(CITATION_RESULT_PATH, citation)
    failure = _failure_analysis(
        result.get("records", []),
        status="completed" if result.get("live_validation") else "blocked",
        reason=result.get("blocked_reason"),
    )
    _write_json(FAILURE_RESULT_PATH, failure)
    _write_json(COMPARISON_RESULT_PATH, _mock_comparison())

    print(
        json.dumps(
            {
                "status": result.get("status"),
                "live_validation": result.get("live_validation"),
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "question_count": len(_questions()),
                "index": _index_stats(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
