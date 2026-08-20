from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.services.embeddings import embed_query
from app.services.vector_store import create_vector_store

logger = logging.getLogger("your_own_ai.retriever")


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot be completed."""


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "text": self.text,
        }


class QueryRetriever:
    """Semantic retrieval layer built on the configured embedding model and FAISS index."""

    def __init__(self, index_path: str | None = None, top_k: int | None = None) -> None:
        self.index_path = index_path or settings.faiss_index_path
        self.top_k = int(top_k if top_k is not None else settings.top_k)
        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than 0.")
        self.vector_store = create_vector_store(index_path=self.index_path, top_k=self.top_k)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        if question is None or not str(question).strip():
            raise ValueError("Question cannot be empty.")

        effective_top_k = int(top_k if top_k is not None else self.top_k)
        if effective_top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        started_at = time.perf_counter()

        try:
            query_vector = embed_query(str(question).strip())
        except Exception as exc:  # pragma: no cover - defensive wrapper
            logger.exception("Failed to generate query embedding for question: %s", question)
            raise RetrievalError("Failed to generate the query embedding.") from exc

        if self.vector_store.is_empty:
            logger.info("No indexed documents available for retrieval.")
            return []

        try:
            raw_results = self.vector_store.search(query_vector, top_k=effective_top_k)
        except ValueError as exc:
            logger.exception("Vector store search failed for question: %s", question)
            raise RetrievalError("Unable to perform vector search with the configured index.") from exc

        retrieval_results: list[RetrievalResult] = []
        for rank, result in enumerate(raw_results, start=1):
            metadata = result.metadata or {}
            retrieval_results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(result.score),
                    document_id=str(metadata.get("document_id", "unknown-document")),
                    filename=str(metadata.get("filename", "unknown-file")),
                    chunk_id=str(metadata.get("chunk_id", f"chunk-{rank}")),
                    page_number=metadata.get("page_number"),
                    text=str(metadata.get("text", "")),
                )
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Retrieval completed in %.2f ms for question: %s | matched=%s",
            elapsed_ms,
            question,
            len(retrieval_results),
        )
        return retrieval_results


retriever = QueryRetriever(index_path=settings.faiss_index_path, top_k=settings.top_k)


def perform_retrieval(question: str, top_k: int | None = None) -> list[RetrievalResult]:
    return retriever.retrieve(question, top_k=top_k)
