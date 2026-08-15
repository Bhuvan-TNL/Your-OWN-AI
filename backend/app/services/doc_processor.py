from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentValidationError(ValueError):
    """Base error for invalid documents."""


class UnsupportedFileTypeError(DocumentValidationError):
    """Raised when a document type is not supported."""


class CorruptedDocumentError(DocumentValidationError):
    """Raised when a document cannot be decoded or parsed."""


class EmptyDocumentError(DocumentValidationError):
    """Raised when a document contains no usable text."""


@dataclass(frozen=True)
class PageContent:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    filename: str
    file_type: str
    text: str
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_text(raw_text: str) -> str:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n +", "\n", normalized)
    normalized = re.sub(r" +\n", "\n", normalized)
    normalized = re.sub(r"\n{2,}", "\n\n", normalized)
    return normalized.strip()


def validate_document_filename(filename: str | None) -> str:
    if not filename:
        raise UnsupportedFileTypeError("A filename is required for document ingestion.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or 'unknown'}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    return filename


def _decode_txt_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise CorruptedDocumentError("TXT document could not be decoded as text.")


def _extract_pdf_text(file_bytes: bytes, filename: str) -> list[PageContent]:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages: list[PageContent] = []
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                cleaned = _normalize_text(extracted)
                if cleaned:
                    pages.append(PageContent(page_number=page.page_number, text=cleaned))
            if not pages:
                raise EmptyDocumentError(f"The PDF '{filename}' is empty or does not contain readable text.")
            return pages
    except DocumentValidationError:
        raise
    except Exception as exc:  # pragma: no cover - broad guard for corrupted PDFs
        raise CorruptedDocumentError(f"The PDF '{filename}' is corrupted or unreadable.") from exc


def _extract_txt_text(file_bytes: bytes, filename: str) -> list[PageContent]:
    try:
        decoded_text = _decode_txt_bytes(file_bytes)
    except CorruptedDocumentError as exc:
        raise CorruptedDocumentError(f"The TXT document '{filename}' could not be decoded.") from exc

    cleaned = _normalize_text(decoded_text)
    if not cleaned:
        raise EmptyDocumentError(f"The TXT document '{filename}' is empty.")

    return [PageContent(page_number=1, text=cleaned)]


def extract_text_from_bytes(file_bytes: bytes, *, filename: str) -> ExtractedDocument:
    validated_name = validate_document_filename(filename)
    if not file_bytes or not file_bytes.strip():
        raise EmptyDocumentError(f"The document '{validated_name}' is empty.")

    suffix = Path(validated_name).suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf_text(file_bytes, validated_name)
        file_type = "pdf"
    elif suffix == ".txt":
        pages = _extract_txt_text(file_bytes, validated_name)
        file_type = "txt"
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    text = "\n\n".join(page.text for page in pages)
    cleaned_text = _normalize_text(text)
    if not cleaned_text:
        raise EmptyDocumentError(f"The document '{validated_name}' is empty after text extraction.")

    return ExtractedDocument(
        document_id="doc-unknown",
        filename=validated_name,
        file_type=file_type,
        text=cleaned_text,
        pages=pages,
        metadata={
            "source_type": file_type,
            "page_count": len(pages),
            "filename": validated_name,
        },
    )


def process_document(file_bytes: bytes, *, filename: str, document_id: str | None = None) -> ExtractedDocument:
    extracted = extract_text_from_bytes(file_bytes, filename=filename)
    final_document_id = document_id or extracted.filename
    return ExtractedDocument(
        document_id=final_document_id,
        filename=extracted.filename,
        file_type=extracted.file_type,
        text=extracted.text,
        pages=extracted.pages,
        metadata={**extracted.metadata, "document_id": final_document_id},
    )
