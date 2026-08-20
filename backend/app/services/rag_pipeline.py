from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.core.config import settings
from app.services.llm import LLMService
from app.services.retriever import QueryRetriever, RetrievalResult

logger = logging.getLogger("your_own_ai.rag_pipeline")


@dataclass(frozen=True)
class SourceReference:
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    rank: int | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "rank": self.rank,
            "score": self.score,
        }


@dataclass(frozen=True)
class RetrievalTiming:
    retrieval_ms: float
    generation_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    timing: RetrievalTiming = field(default_factory=lambda: RetrievalTiming(0.0, 0.0, 0.0))
    provider: str = "mock"
    model: str = "mock-model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "results": self.results,
            "timing": self.timing.to_dict(),
            "provider": self.provider,
            "model": self.model,
        }


class RAGPipeline:
    """Coordinate retrieval and generation into a single grounded answer pipeline."""

    def __init__(
        self,
        *,
        retriever: QueryRetriever | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.retriever = retriever or QueryRetriever(index_path=settings.faiss_index_path, top_k=settings.top_k)
        self.llm_service = llm_service or LLMService(
            provider=settings.llm_provider,
            model=settings.llm_model,
            token=settings.hf_token,
        )

    def build_context(self, results: Sequence[RetrievalResult]) -> str:
        if not results:
            return ""

        snippets: list[str] = []
        for result in results:
            page_label = result.page_number if result.page_number is not None else "unknown"
            snippet = (result.text or "").strip()
            if not snippet:
                continue
            snippets.append(
                f"[Source: {result.filename} | Page: {page_label} | Chunk: {result.chunk_id}]\n{snippet}"
            )
        return "\n\n".join(snippets)

    def execute(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        clean_question = (question or "").strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        retrieval_started = time.perf_counter()
        retrieval_results = self.retriever.retrieve(clean_question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        if not retrieval_results:
            logger.info("No relevant retrieval results found for question: %s", clean_question)
            return RAGResponse(
                question=clean_question,
                answer="I could not find enough relevant information in the uploaded knowledge base to answer this question.",
                sources=[],
                results=[],
                timing=RetrievalTiming(retrieval_ms=retrieval_ms, generation_ms=0.0, total_ms=retrieval_ms),
                provider=self.llm_service.provider,
                model=self.llm_service.model,
            )

        context = self.build_context(retrieval_results)
        generation_started = time.perf_counter()
        llm_result = self.llm_service.generate_answer(clean_question, context)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        answer = llm_result.answer.strip()
        if not answer:
            raise RuntimeError("The configured LLM provider returned an empty answer.")

        sources = [
            SourceReference(
                document_id=result.document_id,
                filename=result.filename,
                chunk_id=result.chunk_id,
                page_number=result.page_number,
                rank=result.rank,
                score=result.score,
            )
            for result in retrieval_results
        ]
        total_ms = retrieval_ms + generation_ms

        return RAGResponse(
            question=clean_question,
            answer=answer,
            sources=sources,
            results=[result.to_dict() for result in retrieval_results],
            timing=RetrievalTiming(retrieval_ms=retrieval_ms, generation_ms=generation_ms, total_ms=total_ms),
            provider=self.llm_service.provider,
            model=self.llm_service.model,
        )


def answer_question(question: str, *, top_k: int | None = None) -> RAGResponse:
    return RAGPipeline().execute(question, top_k=top_k)
