from __future__ import annotations

import numpy as np

from app.services.chunker import DocumentChunk
from app.services.embeddings import embed_query, embed_texts, get_embedding_dimension
from app.services.vector_store import VectorStore


def test_embedding_generation_returns_expected_shape() -> None:
    vectors = embed_texts([
        "Normalization reduces redundancy in a database.",
        "Indexes improve query speed for large tables.",
    ])

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == get_embedding_dimension()
    assert vectors.dtype == np.float32


def test_empty_embedding_input_is_handled() -> None:
    vectors = embed_texts(["", "   ", None])
    assert vectors.shape == (0, get_embedding_dimension())


def test_query_embedding_works() -> None:
    vector = embed_query("How can we avoid repeated data in a database?")
    assert vector.shape == (get_embedding_dimension(),)
    assert vector.dtype == np.float32


def test_faiss_index_can_be_created_and_used(tmp_path) -> None:
    store = VectorStore(index_path=tmp_path / "vector_store", top_k=2)
    assert store.index is not None
    assert store.is_empty is True

    chunks = [
        DocumentChunk(
            document_id="doc-1",
            filename="database_notes.txt",
            chunk_id="doc-1-chunk-0001",
            page_number=1,
            text="Normalization reduces redundancy and improves database integrity.",
        ),
        DocumentChunk(
            document_id="doc-2",
            filename="performance_notes.txt",
            chunk_id="doc-2-chunk-0001",
            page_number=1,
            text="Indexes help retrieve records quickly and efficiently.",
        ),
    ]
    vectors = embed_texts([chunk.text for chunk in chunks])
    store.add_chunks(chunks, vectors)

    assert store.index.ntotal == 2
    results = store.search(vectors[0], top_k=2)
    assert len(results) >= 1
    assert results[0].metadata["chunk_id"] == "doc-1-chunk-0001"


def test_similarity_search_returns_semantically_related_results(tmp_path) -> None:
    store = VectorStore(index_path=tmp_path / "semantic_store", top_k=2)

    chunk_a = DocumentChunk(
        document_id="doc-1",
        filename="dbms.txt",
        chunk_id="doc-1-chunk-0001",
        page_number=1,
        text="Normalization reduces redundancy and improves database integrity.",
    )
    chunk_b = DocumentChunk(
        document_id="doc-2",
        filename="dbms.txt",
        chunk_id="doc-2-chunk-0001",
        page_number=1,
        text="The query optimizer chooses the best execution plan for a database request.",
    )

    vectors = embed_texts([chunk_a.text, chunk_b.text])
    store.add_chunks([chunk_a, chunk_b], vectors)

    query_vector = embed_query("How can we avoid repeated data in a database?")
    results = store.search(query_vector, top_k=2)

    assert results[0].metadata["document_id"] == "doc-1"
    assert "redundancy" in results[0].metadata["text"].lower()


def test_metadata_mapping_matches_vector_positions(tmp_path) -> None:
    store = VectorStore(index_path=tmp_path / "metadata_store", top_k=3)
    chunks = [
        DocumentChunk("doc-x", "file1.txt", "doc-x-chunk-0001", 1, "Alpha beta gamma."),
        DocumentChunk("doc-y", "file2.txt", "doc-y-chunk-0001", 2, "Omega delta epsilon."),
    ]
    vectors = embed_texts([chunk.text for chunk in chunks])
    store.add_chunks(chunks, vectors)

    results = store.search(vectors[0], top_k=2)
    matched = results[0].metadata
    assert matched["document_id"] == "doc-x"
    assert matched["filename"] == "file1.txt"
    assert matched["chunk_id"] == "doc-x-chunk-0001"


def test_index_can_be_saved_and_loaded_again(tmp_path) -> None:
    index_dir = tmp_path / "persisted_index"
    store = VectorStore(index_path=index_dir, top_k=2)
    chunks = [
        DocumentChunk("doc-a", "a.txt", "doc-a-chunk-0001", 1, "Normalization avoids data duplication."),
        DocumentChunk("doc-b", "b.txt", "doc-b-chunk-0001", 1, "Indexes speed up retrieval."),
    ]
    vectors = embed_texts([chunk.text for chunk in chunks])
    store.add_chunks(chunks, vectors)
    store.save()

    reloaded = VectorStore(index_path=index_dir, top_k=2)
    assert reloaded.index is not None
    assert reloaded.index.ntotal == 2
    assert len(reloaded.metadata) == 2

    query_vector = embed_query("How can we reduce repeated information in a database?")
    results = reloaded.search(query_vector, top_k=2)
    assert results[0].metadata["document_id"] == "doc-a"
