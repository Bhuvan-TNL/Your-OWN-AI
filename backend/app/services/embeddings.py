from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger("your_own_ai.embeddings")


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer with a lazy singleton model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embedding_dimension(self) -> int:
        if self._dimension is None:
            try:
                self._dimension = int(self.model.get_sentence_embedding_dimension())
            except AttributeError:
                sample_vector = self.embed_texts(["sample text"])
                self._dimension = int(sample_vector.shape[1])
        logger.debug("Embedding dimension is %s", self._dimension)
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if texts is None:
            raise ValueError("texts cannot be None.")

        cleaned_texts = [str(text).strip() for text in texts if text is not None and str(text).strip()]
        if not cleaned_texts:
            logger.warning("Embedding requested with no non-empty input.")
            return np.empty((0, self.get_embedding_dimension()), dtype=np.float32)

        vectors = self.model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.ndim == 1:
            vector_array = vector_array.reshape(1, -1)
        return vector_array

    def embed_query(self, text: str) -> np.ndarray:
        if text is None or not str(text).strip():
            raise ValueError("Query text cannot be empty.")
        vector = self.embed_texts([text])
        if vector.size == 0:
            raise EmbeddingError("Query embedding produced no vectors.")
        return vector[0]


embedding_model = EmbeddingModel(settings.embedding_model)


def get_embedding_dimension() -> int:
    return embedding_model.get_embedding_dimension()


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    return embedding_model.embed_texts(texts)


def embed_query(text: str) -> np.ndarray:
    return embedding_model.embed_query(text)
