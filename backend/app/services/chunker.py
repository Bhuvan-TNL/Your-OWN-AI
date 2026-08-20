from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.core.config import settings


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None
    text: str
    content_hash: str | None = None


def validate_chunk_parameters(chunk_size: int | None = None, chunk_overlap: int | None = None) -> tuple[int, int]:
    effective_chunk_size = int(chunk_size if chunk_size is not None else settings.chunk_size)
    effective_chunk_overlap = int(chunk_overlap if chunk_overlap is not None else settings.chunk_overlap)

    if effective_chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if effective_chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if effective_chunk_overlap >= effective_chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    return effective_chunk_size, effective_chunk_overlap


def _normalize_chunk_text(raw_text: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_text).strip()
    return normalized


def chunk_text(
    text: str,
    *,
    document_id: str,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    page_number: int | None = None,
) -> list[DocumentChunk]:
    effective_chunk_size, effective_overlap = validate_chunk_parameters(chunk_size, chunk_overlap)
    cleaned_text = _normalize_chunk_text(text)
    if not cleaned_text:
        return []

    words = cleaned_text.split()
    if len(words) <= effective_chunk_size:
        return [
            DocumentChunk(
                document_id=document_id,
                filename=filename,
                chunk_id=f"{document_id}-chunk-0001",
                page_number=page_number,
                text=cleaned_text,
            )
        ]

    step = max(effective_chunk_size - effective_overlap, 1)
    chunks: list[DocumentChunk] = []
    for index in range(0, len(words), step):
        window = words[index : index + effective_chunk_size]
        if not window:
            continue
        chunk_text = " ".join(window)
        chunks.append(
            DocumentChunk(
                document_id=document_id,
                filename=filename,
                chunk_id=f"{document_id}-chunk-{len(chunks) + 1:04d}",
                page_number=page_number,
                text=chunk_text,
            )
        )
        if len(window) < effective_chunk_size:
            break

    return chunks


def chunk_document(
    document_text: str,
    *,
    document_id: str,
    filename: str,
    page_numbers: list[int] | None = None,
    page_content: Sequence[tuple[int, str]] | None = None,
    content_hash: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    effective_chunk_size, effective_overlap = validate_chunk_parameters(chunk_size, chunk_overlap)
    word_records: list[tuple[str, int | None]] = []

    if page_content is not None:
        for page_number, page_text in page_content:
            cleaned_page_text = _normalize_chunk_text(page_text)
            word_records.extend((word, page_number) for word in cleaned_page_text.split())
    else:
        cleaned_text = _normalize_chunk_text(document_text)
        fallback_page_number = (page_numbers or [1])[0]
        word_records.extend((word, fallback_page_number) for word in cleaned_text.split())

    if not word_records:
        return []

    step = max(effective_chunk_size - effective_overlap, 1)
    chunks: list[DocumentChunk] = []
    for start_index in range(0, len(word_records), step):
        window = word_records[start_index : start_index + effective_chunk_size]
        if not window:
            continue

        chunks.append(
            DocumentChunk(
                document_id=document_id,
                filename=filename,
                chunk_id=f"{document_id}-chunk-{len(chunks) + 1:04d}",
                page_number=window[0][1],
                text=" ".join(word for word, _ in window),
                content_hash=content_hash,
            )
        )
        if len(window) < effective_chunk_size:
            break

    return chunks
