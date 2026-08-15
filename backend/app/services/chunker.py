from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None
    text: str


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
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    effective_chunk_size, effective_overlap = validate_chunk_parameters(chunk_size, chunk_overlap)
    cleaned_text = _normalize_chunk_text(document_text)
    if not cleaned_text:
        return []

    page_numbers = page_numbers or [1]
    page_sections = cleaned_text.split("\n\n")
    if len(page_sections) == 1 and len(page_numbers) > 1:
        page_sections = [cleaned_text]

    chunks: list[DocumentChunk] = []
    for page_index, page_text in enumerate(page_sections):
        page_number = page_numbers[page_index] if page_index < len(page_numbers) else page_numbers[0]
        chunks.extend(
            chunk_text(
                page_text,
                document_id=document_id,
                filename=filename,
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_overlap,
                page_number=page_number,
            )
        )

    final_chunks: list[DocumentChunk] = []
    for index, chunk in enumerate(chunks, start=1):
        final_chunks.append(
            DocumentChunk(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_id=f"{document_id}-chunk-{index:04d}",
                page_number=chunk.page_number,
                text=chunk.text,
            )
        )

    return final_chunks
