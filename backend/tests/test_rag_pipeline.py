from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import query as query_module
from app.main import app
from app.services.chunker import DocumentChunk
from app.services.embeddings import embed_texts
from app.services.llm import LLMService
from app.services.llm import LLMProviderError
from app.services.rag_pipeline import RAGPipeline
from app.services.retriever import QueryRetriever, RetrievalResult
from app.services.vector_store import VectorStore


def _index_sample_documents(index_path) -> None:
    chunks = [
        DocumentChunk(
            document_id="dbms-1",
            filename="DBMS_Unit1.pdf",
            chunk_id="dbms-1-chunk-0001",
            page_number=14,
            text="Normalization reduces redundancy and improves database integrity.",
        ),
        DocumentChunk(
            document_id="dbms-2",
            filename="DBMS_Unit2.pdf",
            chunk_id="dbms-2-chunk-0001",
            page_number=3,
            text="Indexes improve query speed for large tables.",
        ),
    ]
    vectors = embed_texts([chunk.text for chunk in chunks])
    store = VectorStore(index_path=index_path, top_k=5)
    store.add_chunks(chunks, vectors)
    store.save()


def test_rag_pipeline_returns_grounded_answer(tmp_path) -> None:
    index_path = tmp_path / "rag_index"
    _index_sample_documents(index_path)

    pipeline = RAGPipeline(
        retriever=QueryRetriever(index_path=str(index_path), top_k=2),
        llm_service=LLMService(provider="mock", model="mock-model"),
    )

    response = pipeline.execute("How can repeated data be reduced in a database?")

    assert response.question == "How can repeated data be reduced in a database?"
    assert response.answer
    assert response.sources
    assert response.results
    assert response.timing.total_ms >= response.timing.retrieval_ms
    assert response.sources[0].filename == "DBMS_Unit1.pdf"


def test_rag_pipeline_handles_missing_index(tmp_path) -> None:
    pipeline = RAGPipeline(
        retriever=QueryRetriever(index_path=str(tmp_path / "missing_index"), top_k=2),
        llm_service=LLMService(provider="mock", model="mock-model"),
    )

    response = pipeline.execute("What is normalization?")

    assert response.sources == []
    assert "could not find enough relevant information" in response.answer.lower()
    assert response.timing.generation_ms == 0.0


def test_query_api_returns_rag_response(monkeypatch) -> None:
    class FakeSource:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeTiming:
        def __init__(self, retrieval_ms: float, generation_ms: float, total_ms: float) -> None:
            self.retrieval_ms = retrieval_ms
            self.generation_ms = generation_ms
            self.total_ms = total_ms

        def to_dict(self) -> dict[str, float]:
            return {
                "retrieval_ms": self.retrieval_ms,
                "generation_ms": self.generation_ms,
                "total_ms": self.total_ms,
            }

    class FakePipeline:
        def execute(self, question: str):
            return SimpleNamespace(
                question=question,
                answer="Grounded answer based on the available context.",
                sources=[
                    FakeSource(
                        document_id="doc-1",
                        filename="DBMS_Unit3.pdf",
                        chunk_id="doc-1-chunk-0001",
                        page_number=12,
                        rank=1,
                        score=0.9,
                    )
                ],
                results=[
                    {
                        "rank": 1,
                        "score": 0.9,
                        "document_id": "doc-1",
                        "filename": "DBMS_Unit3.pdf",
                        "chunk_id": "doc-1-chunk-0001",
                        "page_number": 12,
                        "text": "Normalization reduces redundancy.",
                    }
                ],
                timing=FakeTiming(20.0, 50.0, 70.0),
                provider="mock",
                model="mock-model",
            )

    monkeypatch.setattr(query_module, "RAGPipeline", FakePipeline)

    client = TestClient(app)
    response = client.post("/api/query", json={"question": "What is normalization?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What is normalization?"
    assert payload["answer"] == "Grounded answer based on the available context."
    assert payload["sources"][0]["filename"] == "DBMS_Unit3.pdf"
    assert payload["results"][0]["rank"] == 1
    assert payload["timing"]["total_ms"] == 70.0


def test_context_formatting_preserves_source_metadata() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    results = [
        RetrievalResult(
            rank=1,
            score=0.91,
            document_id="doc-1",
            filename="Cloud Computing Notes.pdf",
            chunk_id="doc-1-chunk-0003",
            page_number=96,
            text="SOA exposes loosely coupled services through standard interfaces.",
        )
    ]

    context = pipeline.build_context(results)

    assert "Cloud Computing Notes.pdf" in context
    assert "Page: 96" in context
    assert "Chunk: doc-1-chunk-0003" in context
    assert "loosely coupled services" in context


def test_pipeline_passes_context_to_mock_llm_and_preserves_sources() -> None:
    class StaticRetriever:
        def retrieve(self, question: str, top_k: int | None = None):
            return [
                RetrievalResult(
                    rank=1,
                    score=0.88,
                    document_id="doc-1",
                    filename="notes.txt",
                    chunk_id="doc-1-chunk-0001",
                    page_number=4,
                    text="Service-oriented architecture uses loosely coupled services.",
                )
            ]

    pipeline = RAGPipeline(
        retriever=StaticRetriever(),
        llm_service=LLMService(provider="mock", model="mock-model"),
    )

    response = pipeline.execute("What does SOA use?")

    assert "loosely coupled services" in response.answer
    assert response.sources[0].document_id == "doc-1"
    assert response.sources[0].page_number == 4
    assert response.sources[0].score == 0.88
    assert response.timing.generation_ms >= 0


def test_pipeline_propagates_provider_failure() -> None:
    class StaticRetriever:
        def retrieve(self, question: str, top_k: int | None = None):
            return [
                RetrievalResult(1, 0.9, "doc-1", "notes.txt", "chunk-1", 1, "Relevant context.")
            ]

    class FailingLLM:
        provider = "huggingface"
        model = "test-model"

        def generate_answer(self, question: str, context: str):
            raise LLMProviderError("provider unavailable")

    pipeline = RAGPipeline(retriever=StaticRetriever(), llm_service=FailingLLM())

    with pytest.raises(LLMProviderError, match="provider unavailable"):
        pipeline.execute("Question")


def test_query_api_reports_provider_failure(monkeypatch) -> None:
    class FailingPipeline:
        def execute(self, question: str):
            raise LLMProviderError("provider unavailable")

    monkeypatch.setattr(query_module, "RAGPipeline", FailingPipeline)

    response = TestClient(app).post("/api/query", json={"question": "Question"})

    assert response.status_code == 503
    assert "provider unavailable" in response.json()["detail"]
