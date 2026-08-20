from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import BACKEND_DIR, settings
from app.services.chunker import chunk_document
from app.services.doc_processor import (
    CorruptedDocumentError,
    DocumentValidationError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    process_document,
)
from app.services.embeddings import embed_texts
from app.services.vector_store import create_vector_store

router = APIRouter(prefix="/api", tags=["ingestion"])
RAW_DOCUMENTS_DIR = BACKEND_DIR / "data" / "raw_documents"
RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _find_raw_document_by_hash(content_hash: str) -> Path | None:
    for candidate in RAW_DOCUMENTS_DIR.iterdir():
        if not candidate.is_file():
            continue
        try:
            candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
        if candidate_hash == content_hash:
            return candidate
    return None


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")

    contents = await file.read()
    if not contents or not contents.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file '{file.filename}' is empty.",
        )

    safe_filename = Path(file.filename).name
    stored_name = f"{Path(safe_filename).stem}-{uuid.uuid4().hex}{Path(safe_filename).suffix.lower()}"
    stored_path = RAW_DOCUMENTS_DIR / stored_name
    document_id = f"{Path(safe_filename).stem}-{uuid.uuid4().hex}"
    content_hash = hashlib.sha256(contents).hexdigest()

    try:
        processed = process_document(contents, filename=file.filename, document_id=document_id)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except CorruptedDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    chunked = chunk_document(
        processed.text,
        document_id=processed.document_id,
        filename=processed.filename,
        page_numbers=[page.page_number for page in processed.pages],
        page_content=[(page.page_number, page.text) for page in processed.pages],
        content_hash=content_hash,
    )

    if not chunked:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No usable content was generated from the uploaded document.")

    try:
        vector_store = create_vector_store(index_path=settings.faiss_index_path, top_k=settings.top_k)

        existing_document_ids = vector_store.find_document_ids_by_content_hash(content_hash)
        if existing_document_ids:
            existing_document_id = existing_document_ids[0]
            existing_chunks = [
                metadata
                for metadata in vector_store.metadata
                if metadata.get("document_id") == existing_document_id
            ]
            existing_raw_path = _find_raw_document_by_hash(content_hash)
            if existing_raw_path is None:
                stored_path.write_bytes(contents)
                existing_raw_path = stored_path
            return {
                "message": "Document was already indexed; no duplicate vectors were added.",
                "duplicate": True,
                "document_id": existing_document_id,
                "filename": processed.filename,
                "stored_path": str(existing_raw_path),
                "file_type": processed.file_type,
                "page_count": len({metadata.get("page_number") for metadata in existing_chunks}),
                "chunk_count": len(existing_chunks),
                "chunks": [
                    {
                        "chunk_id": metadata.get("chunk_id"),
                        "page_number": metadata.get("page_number"),
                        "text": metadata.get("text", ""),
                    }
                    for metadata in existing_chunks
                ],
            }

        chunk_vectors = embed_texts([chunk.text for chunk in chunked])
        stored_path.write_bytes(contents)
        vector_store.add_chunks(chunked, chunk_vectors)
        vector_store.save()
    except Exception as exc:  # pragma: no cover - guard against embedding/index failures during ingestion
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document could not be indexed: {exc}",
        ) from exc

    return {
        "message": "Document ingested successfully.",
        "document_id": processed.document_id,
        "filename": processed.filename,
        "stored_path": str(stored_path),
        "file_type": processed.file_type,
        "page_count": len(processed.pages),
        "chunk_count": len(chunked),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
            }
            for chunk in chunked
        ],
    }
