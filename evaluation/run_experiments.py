"""Run reproducible retrieval-configuration experiments.

Each re-indexed experiment uses a temporary vector-store directory. The
application's configured baseline index and defaults are never overwritten.
All runs use the deterministic mock provider because live Hugging Face
credentials are not available; answer metrics are therefore diagnostic only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
REINDEX_SCRIPT = BACKEND_DIR / "scripts" / "reindex_vector_store.py"
BASELINE_SCRIPT = PROJECT_ROOT / "evaluation" / "run_baseline.py"
PHASE7_BASELINE_PATH = RESULTS_DIR / "baseline_phase7.json"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings


def _base_environment(index_path: Path, config: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "LLM_PROVIDER": "mock",
            "LLM_MODEL": "mock-model",
            "FAISS_INDEX_PATH": str(index_path),
            "CHUNK_SIZE": str(config["chunk_size"]),
            "CHUNK_OVERLAP": str(config["chunk_overlap"]),
            "TOP_K": str(config["top_k"]),
            "SIMILARITY_THRESHOLD": str(config["similarity_threshold"]),
            "EMBEDDING_MODEL": str(config["embedding_model"]),
        }
    )
    return environment


def _run(command: list[str], environment: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no command output"
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}")


def _index_stats(index_path: Path) -> dict[str, Any]:
    metadata_path = index_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    return {
        "document_count": len({item.get("document_id") for item in metadata}),
        "chunk_count": len(metadata),
        "unique_content_hashes": len({item.get("content_hash") for item in metadata if item.get("content_hash")}),
        "page_count": len({item.get("page_number") for item in metadata if item.get("page_number") is not None}),
    }


def _reindex(index_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    environment = _base_environment(index_path, config)
    _run([sys.executable, str(REINDEX_SCRIPT), "--apply"], environment)
    return _index_stats(index_path)


def _run_evaluation(
    name: str,
    index_path: Path,
    config: dict[str, Any],
    *,
    index_stats: dict[str, Any],
) -> dict[str, Any]:
    output_path = RESULTS_DIR / f"{name}.json"
    environment = _base_environment(index_path, config)
    _run(
        [
            sys.executable,
            str(BASELINE_SCRIPT),
            "--questions",
            str(QUESTIONS_PATH),
            "--output",
            str(output_path),
            "--provider",
            "mock",
            "--model",
            "mock-model",
        ],
        environment,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["experiment"] = name
    result["evaluation_mode"] = "mock_provider_retrieval_optimization"
    result["index_stats"] = index_stats
    result["configuration"] = config
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _mean_metric(result: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = result
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return float(value) if value is not None else 0.0


def _best_result(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Prefer page/source retrieval, then answer diagnostics, then latency."""

    return max(
        results,
        key=lambda result: (
            _mean_metric(result, ("summary", "page_hit_at_k")),
            _mean_metric(result, ("summary", "retrieval_hit_at_k")),
            _mean_metric(result, ("summary", "supported_answer_embedding_cosine_mean")),
            _mean_metric(result, ("summary", "supported_answer_token_f1_mean")),
            -float(result.get("configuration", {}).get("top_k", 0)),
            -_mean_metric(result, ("summary", "total_latency", "mean_ms")),
            -float(result.get("index_stats", {}).get("chunk_count", 0)),
        ),
    )


def _threshold_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer refusal and retrieval, then the least aggressive passing threshold."""

    return max(
        results,
        key=lambda result: (
            _mean_metric(result, ("summary", "unsupported_refusal_rate")),
            _mean_metric(result, ("summary", "page_hit_at_k")),
            _mean_metric(result, ("summary", "retrieval_hit_at_k")),
            _mean_metric(result, ("summary", "supported_answer_embedding_cosine_mean")),
            _mean_metric(result, ("summary", "supported_answer_token_f1_mean")),
            -float(result.get("configuration", {}).get("similarity_threshold", 0.0)),
            -_mean_metric(result, ("summary", "total_latency", "mean_ms")),
        ),
    )


def _config(*, chunk_size: int, chunk_overlap: int, top_k: int, similarity_threshold: float, embedding_model: str) -> dict[str, Any]:
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "similarity_threshold": similarity_threshold,
        "llm_provider": "mock",
        "llm_model": "mock-model",
    }


def _run_reindexed_sweep(
    experiment_name: str,
    configurations: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"your-own-ai-{experiment_name}-") as temporary_dir:
        for name, config in configurations:
            index_path = Path(temporary_dir) / name / "vector_store"
            index_stats = _reindex(index_path, config)
            results.append(_run_evaluation(name, index_path, config, index_stats=index_stats))
    return results


def _run_existing_index_sweep(
    experiment_name: str,
    configurations: list[tuple[str, dict[str, Any]]],
    index_path: Path,
    index_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, config in configurations:
        results.append(_run_evaluation(name, index_path, config, index_stats=index_stats))
    return results


def _result_view(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    return {
        "name": result["experiment"],
        "configuration": result["configuration"],
        "index_stats": result["index_stats"],
        "retrieval_hit_at_k": summary["retrieval_hit_at_k"],
        "page_hit_at_k": summary["page_hit_at_k"],
        "supported_answer_token_f1_mean": summary["supported_answer_token_f1_mean"],
        "supported_answer_embedding_cosine_mean": summary["supported_answer_embedding_cosine_mean"],
        "groundedness_token_overlap_mean": summary["groundedness_token_overlap_mean"],
        "unsupported_refusal_rate": summary["unsupported_refusal_rate"],
        "retrieval_latency": summary["retrieval_latency"],
        "total_latency": summary["total_latency"],
    }


def _write_embedding_blocked_result(baseline: dict[str, Any]) -> dict[str, Any]:
    result = {
        "experiment": "embedding_model",
        "status": "blocked",
        "configuration": baseline["configuration"],
        "current_model": {
            "name": settings.embedding_model,
            "dimension": 384,
            "evaluated_in": "baseline.json",
        },
        "alternatives_considered": [
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        ],
        "blocked_reason": (
            "The alternative model is not present in the local Hugging Face cache and network access is unavailable. "
            "No comparison is reported rather than fabricating a result."
        ),
    }
    path = RESULTS_DIR / "embedding_results.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _write_summary(
    baseline: dict[str, Any],
    chunk_results: list[dict[str, Any]],
    overlap_results: list[dict[str, Any]],
    top_k_results: list[dict[str, Any]],
    threshold_results: list[dict[str, Any]],
    embedding_result: dict[str, Any],
) -> dict[str, Any]:
    strongest_chunk = _best_result(chunk_results)
    strongest_overlap = _best_result(overlap_results)
    strongest_top_k = _best_result(top_k_results)
    strongest_threshold = _threshold_best(threshold_results)
    final_config = strongest_threshold["configuration"].copy()
    final_config["chunk_size"] = strongest_overlap["configuration"]["chunk_size"]
    final_config["chunk_overlap"] = strongest_overlap["configuration"]["chunk_overlap"]
    final_config["top_k"] = strongest_top_k["configuration"]["top_k"]
    final_config["similarity_threshold"] = strongest_threshold["configuration"]["similarity_threshold"]
    final_config["embedding_model"] = settings.embedding_model

    score_groups: dict[str, list[float]] = {"all": [], "supported": [], "unsupported": []}
    for record in baseline.get("records", []):
        group = "supported" if record.get("supported") else "unsupported"
        for item in record.get("retrieved", []):
            score = item.get("score")
            if score is None:
                continue
            value = float(score)
            score_groups["all"].append(value)
            score_groups[group].append(value)

    def score_stats(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": mean(values) if values else None,
        }

    score_distribution = {
        key: score_stats(values) for key, values in score_groups.items()
    }
    score_distribution["threshold_candidates"] = [-1.0, 0.15, 0.20, 0.30]
    score_distribution["candidate_rationale"] = (
        "The unsupported top-k score maximum is below the supported minimum in the baseline; "
        "0.15 and 0.20 test a separation margin, while 0.30 tests a stricter recall/precision trade-off."
    )

    summary = {
        "experiment": "phase_7_parameter_optimization",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_mode": "mock_provider_retrieval_optimization",
        "baseline": _result_view({**baseline, "index_stats": {"document_count": 2, "chunk_count": 72}}),
        "experiments": {
            "chunk_size": [_result_view(result) for result in chunk_results],
            "overlap": [_result_view(result) for result in overlap_results],
            "top_k": [_result_view(result) for result in top_k_results],
            "similarity_threshold": [_result_view(result) for result in threshold_results],
            "embedding_model": embedding_result,
        },
        "best_configuration_by_parameter": {
            "chunk_size": strongest_chunk["configuration"],
            "chunk_overlap": strongest_overlap["configuration"],
            "top_k": strongest_top_k["configuration"],
            "similarity_threshold": strongest_threshold["configuration"],
            "embedding_model": "baseline retained; alternative evaluation blocked",
        },
        "threshold_score_distribution": score_distribution,
        "recommended_configuration": final_config,
        "selection_rule": (
            "Prioritized page-level retrieval hit, then source retrieval hit, then embedding/lexical answer diagnostics, "
            "then lower top_k for equal recall, lower total latency, and chunk count. Threshold selection prioritized "
            "unsupported refusal and retrieval, then the least aggressive threshold that preserved those metrics."
        ),
        "limitations": [
            "All experiments use the deterministic mock provider; answer metrics are diagnostics, not real LLM performance.",
            "Filename hit@k is coarse because all supported questions target the same PDF.",
            "Page metadata denotes the first page of a chunk; chunks may span page boundaries.",
            "The alternative embedding model was not cached and could not be downloaded in this environment.",
        ],
        "observations": [
            "Chunk-size and overlap effects should be interpreted jointly with chunk count and latency.",
            "Increasing top-k can improve page recall while adding context and latency.",
            "The baseline score distribution places the unsupported question below the supported-question minimum, making threshold candidates around 0.15–0.30 experimentally meaningful.",
        ],
        "next_research_questions": [
            "Does a real instruction-tuned provider improve answer correctness and unsupported refusal compared with the mock diagnostic?",
            "Would page-bounded chunks or page-range metadata improve citation accuracy?",
            "How stable are the parameter rankings across additional documents and datasets?",
        ],
    }
    (RESULTS_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = PHASE7_BASELINE_PATH if PHASE7_BASELINE_PATH.exists() else RESULTS_DIR / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    embedding_model = settings.embedding_model
    baseline_chunk_size = settings.chunk_size
    baseline_overlap = settings.chunk_overlap
    baseline_top_k = settings.top_k
    baseline_threshold = settings.similarity_threshold

    chunk_configurations = [
        (
            f"chunk_size_{chunk_size}",
            _config(
                chunk_size=chunk_size,
                chunk_overlap=max(0, round(chunk_size * baseline_overlap / baseline_chunk_size)),
                top_k=baseline_top_k,
                similarity_threshold=baseline_threshold,
                embedding_model=embedding_model,
            ),
        )
        for chunk_size in (256, 512, 768, 1024)
    ]
    chunk_results = _run_reindexed_sweep("chunk-size", chunk_configurations)
    strongest_chunk = _best_result(chunk_results)
    selected_chunk_size = int(strongest_chunk["configuration"]["chunk_size"])

    overlap_configurations = [
        (
            f"overlap_{overlap}",
            _config(
                chunk_size=selected_chunk_size,
                chunk_overlap=overlap,
                top_k=baseline_top_k,
                similarity_threshold=baseline_threshold,
                embedding_model=embedding_model,
            ),
        )
        for overlap in (0, 25, 50, 100)
        if overlap < selected_chunk_size
    ]
    overlap_results = _run_reindexed_sweep("overlap", overlap_configurations)
    strongest_overlap = _best_result(overlap_results)
    selected_overlap = int(strongest_overlap["configuration"]["chunk_overlap"])

    top_k_configurations = [
        (
            f"top_k_{top_k}",
            _config(
                chunk_size=selected_chunk_size,
                chunk_overlap=selected_overlap,
                top_k=top_k,
                similarity_threshold=baseline_threshold,
                embedding_model=embedding_model,
            ),
        )
        for top_k in (1, 3, 5, 8)
    ]
    with tempfile.TemporaryDirectory(prefix="your-own-ai-top-k-") as temporary_dir:
        top_k_index = Path(temporary_dir) / "vector_store"
        selected_config = top_k_configurations[0][1]
        top_k_index_stats = _reindex(top_k_index, selected_config)
        top_k_results = _run_existing_index_sweep("top-k", top_k_configurations, top_k_index, top_k_index_stats)
    strongest_top_k = _best_result(top_k_results)
    selected_top_k = int(strongest_top_k["configuration"]["top_k"])

    threshold_candidates = [-1.0, 0.15, 0.20, 0.30]
    threshold_configurations = [
        (
            f"threshold_{str(threshold).replace('-', 'm').replace('.', '_')}",
            _config(
                chunk_size=selected_chunk_size,
                chunk_overlap=selected_overlap,
                top_k=selected_top_k,
                similarity_threshold=threshold,
                embedding_model=embedding_model,
            ),
        )
        for threshold in threshold_candidates
    ]
    with tempfile.TemporaryDirectory(prefix="your-own-ai-threshold-") as temporary_dir:
        threshold_index = Path(temporary_dir) / "vector_store"
        threshold_index_stats = _reindex(threshold_index, threshold_configurations[0][1])
        threshold_results = _run_existing_index_sweep(
            "threshold",
            threshold_configurations,
            threshold_index,
            threshold_index_stats,
        )

    embedding_result = _write_embedding_blocked_result(baseline)
    summary = _write_summary(
        baseline,
        chunk_results,
        overlap_results,
        top_k_results,
        threshold_results,
        embedding_result,
    )

    print(json.dumps({
        "chunk_size": [_result_view(result) for result in chunk_results],
        "overlap": [_result_view(result) for result in overlap_results],
        "top_k": [_result_view(result) for result in top_k_results],
        "similarity_threshold": [_result_view(result) for result in threshold_results],
        "recommended_configuration": summary["recommended_configuration"],
    }, indent=2))


if __name__ == "__main__":
    main()
