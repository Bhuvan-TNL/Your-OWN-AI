from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from app.core.config import settings
from app.services.chunker import DocumentChunk
from app.services.embeddings import get_embedding_dimension

logger = logging.getLogger("your_own_ai.vector_store")


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Wrapper around a FAISS index with persisted chunk metadata."""

    def __init__(self, index_path: str | Path | None = None, top_k: int | None = None) -> None:
        base_path = Path(index_path) if index_path is not None else Path(settings.faiss_index_path)
        self.index_dir = base_path if base_path.suffix == "" else base_path.parent
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss_index.bin"
        self.metadata_path = self.index_dir / "metadata.json"
        self.top_k = int(top_k if top_k is not None else settings.top_k)
        self.dimension = get_embedding_dimension()
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self.index_path.exists():
            logger.info("Loading FAISS index from %s", self.index_path)
            self.index = faiss.read_index(str(self.index_path))
            if self.metadata_path.exists():
                with self.metadata_path.open("r", encoding="utf-8") as file_handle:
                    self.metadata = json.load(file_handle)
            else:
                self.metadata = []
            if self.index.d != self.dimension:
                raise ValueError(
                    f"Existing index dimension {self.index.d} does not match embedding dimension {self.dimension}."
                )
            return

        logger.info("Creating new FAISS index with dimension %s", self.dimension)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        normalized = np.asarray(vectors, dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized.reshape(1, -1)
        if normalized.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: got {normalized.shape[1]} but expected {self.dimension}."
            )
        faiss.normalize_L2(normalized)
        return normalized

    def add_vectors(self, vectors: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")
        if len(metadata) == 0:
            return

        normalized_vectors = self._normalize_vectors(vectors)
        if normalized_vectors.shape[0] != len(metadata):
            raise ValueError("Number of vectors must match the number of metadata entries.")

        logger.info("Adding %s vectors to FAISS index", len(metadata))
        self.index.add(normalized_vectors)
        self.metadata.extend(list(metadata))

    def add_chunks(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk count does not match number of vectors.")

        metadata_list: list[dict[str, Any]] = []
        for chunk in chunks:
            metadata_list.append(
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                }
            )
        self.add_vectors(vectors, metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int | None = None) -> list[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        effective_k = int(top_k if top_k is not None else self.top_k)
        if effective_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        normalized_query = self._normalize_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        k = min(effective_k, self.index.ntotal)
        scores, indices = self.index.search(normalized_query, k)

        results: list[SearchResult] = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist()):
            if index < 0 or index >= len(self.metadata):
                continue
            results.append(
                SearchResult(
                    index=int(index),
                    score=float(score),
                    metadata=self.metadata[int(index)],
                )
            )
        return results

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an uninitialized FAISS index.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.metadata, file_handle, ensure_ascii=False, indent=2)
        logger.info("Saved FAISS index to %s with %s metadata entries", self.index_path, len(self.metadata))

    def load(self) -> None:
        if not self.index_path.exists():
            logger.warning("No FAISS index found at %s. Creating a new empty index.", self.index_path)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            return

        self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as file_handle:
                self.metadata = json.load(file_handle)
        else:
            self.metadata = []

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0


vector_store = VectorStore(index_path=settings.faiss_index_path, top_k=settings.top_k)


def create_vector_store(index_path: str | Path | None = None, top_k: int | None = None) -> VectorStore:
    return VectorStore(index_path=index_path, top_k=top_k)
