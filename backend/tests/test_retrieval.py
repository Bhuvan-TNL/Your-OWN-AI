from __future__ import annotations

import time

import pytest

from app.services.chunker import DocumentChunk
from app.services.embeddings import embed_texts, embed_query
from app.services.retriever import QueryRetriever
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
        DocumentChunk(
            document_id="os-1",
            filename="Operating_Systems.pdf",
            chunk_id="os-1-chunk-0001",
            page_number=9,
            text="Scheduling algorithms decide which process runs next.",
        ),
    ]
    vectors = embed_texts([chunk.text for chunk in chunks])
    store = VectorStore(index_path=index_path, top_k=5)
    store.add_chunks(chunks, vectors)
    store.save()


def test_retriever_returns_ranked_results(tmp_path) -> None:
    index_path = tmp_path / "retrieval_index"
    _index_sample_documents(index_path)

    retriever = QueryRetriever(index_path=str(index_path), top_k=2)
    results = retriever.retrieve("How can repeated data be reduced in a database?", top_k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].filename == "DBMS_Unit1.pdf"
    assert results[0].document_id == "dbms-1"
    assert results[0].page_number == 14
    assert "redundancy" in results[0].text.lower()
    assert results[0].score >= results[1].score


def test_top_k_is_respected(tmp_path) -> None:
    index_path = tmp_path / "topk_index"
    _index_sample_documents(index_path)

    retriever = QueryRetriever(index_path=str(index_path), top_k=5)
    results = retriever.retrieve("How can data duplication be reduced?", top_k=1)

    assert len(results) == 1


def test_metadata_is_preserved(tmp_path) -> None:
    index_path = tmp_path / "metadata_index"
    _index_sample_documents(index_path)

    retriever = QueryRetriever(index_path=str(index_path), top_k=3)
    results = retriever.retrieve("database integrity and redundancy", top_k=3)

    assert results[0].chunk_id
    assert results[0].document_id
    assert results[0].filename
    assert results[0].page_number is not None
    assert results[0].text


def test_empty_question_is_rejected(tmp_path) -> None:
    retriever = QueryRetriever(index_path=str(tmp_path / "empty_question_index"), top_k=3)

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_missing_index_is_handled_cleanly(tmp_path) -> None:
    retriever = QueryRetriever(index_path=str(tmp_path / "missing_index"), top_k=3)
    results = retriever.retrieve("Explain the database normalization process.")

    assert results == []


def test_result_structure_is_complete(tmp_path) -> None:
    index_path = tmp_path / "structure_index"
    _index_sample_documents(index_path)

    retriever = QueryRetriever(index_path=str(index_path), top_k=3)
    results = retriever.retrieve("database improvement and repeated data", top_k=3)

    first = results[0]
    assert set(first.to_dict().keys()) == {
        "rank",
        "score",
        "document_id",
        "filename",
        "chunk_id",
        "page_number",
        "text",
    }


def test_query_embedding_is_generated(tmp_path) -> None:
    index_path = tmp_path / "embedding_index"
    _index_sample_documents(index_path)

    retriever = QueryRetriever(index_path=str(index_path), top_k=3)
    start = time.perf_counter()
    results = retriever.retrieve("How to avoid repeated data in a database?", top_k=2)
    elapsed = time.perf_counter() - start

    assert len(results) > 0
    assert elapsed >= 0.0


def test_invalid_top_k_is_rejected(tmp_path) -> None:
    retriever = QueryRetriever(index_path=str(tmp_path / "invalid_index"), top_k=3)

    with pytest.raises(ValueError):
        retriever.retrieve("question", top_k=0)


def test_retriever_refreshes_persisted_index_after_upload(tmp_path) -> None:
    index_path = tmp_path / "live_refresh_index"

    sample_chunk = DocumentChunk(
        document_id="sample-doc",
        filename="sample.txt",
        chunk_id="sample-doc-chunk-0001",
        page_number=1,
        text="Alpha beta gamma delta epsilon zeta eta. More content after the break.",
    )
    sample_store = VectorStore(index_path=index_path, top_k=3)
    sample_store.add_chunks([sample_chunk], embed_texts([sample_chunk.text]))
    sample_store.save()

    retriever = QueryRetriever(index_path=str(index_path), top_k=3)
    sample_results = retriever.retrieve("Alpha beta gamma delta epsilon", top_k=3)
    assert sample_results
    assert sample_results[0].filename == "sample.txt"

    soa_chunk = DocumentChunk(
        document_id="cloud-doc",
        filename="Cloud Computing Notes.pdf",
        chunk_id="cloud-doc-chunk-0001",
        page_number=14,
        text="Service-Oriented Architecture (SOA) is an architectural model in which loosely coupled services are exposed over a network so that organizations can access cloud-based services through standard interfaces.",
    )
    live_store = VectorStore(index_path=index_path, top_k=3)
    live_store.add_chunks([soa_chunk], embed_texts([soa_chunk.text]))
    live_store.save()

    refreshed_results = retriever.retrieve("What is Service-Oriented Architecture (SOA)?", top_k=1)
    assert refreshed_results
    assert refreshed_results[0].filename == "Cloud Computing Notes.pdf"
    assert refreshed_results[0].document_id == "cloud-doc"
    assert "SOA" in refreshed_results[0].text or "Service-Oriented Architecture" in refreshed_results[0].text
    assert refreshed_results[0].filename != "sample.txt"
