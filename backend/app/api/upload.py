from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import BACKEND_DIR
from app.services.chunker import chunk_document
from app.services.doc_processor import (
    CorruptedDocumentError,
    DocumentValidationError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    process_document,
)

router = APIRouter(prefix="/api", tags=["ingestion"])
RAW_DOCUMENTS_DIR = BACKEND_DIR / "data" / "raw_documents"
RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


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

    try:
        processed = process_document(contents, filename=file.filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except CorruptedDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    safe_filename = Path(file.filename).name
    stored_name = f"{Path(safe_filename).stem}-{uuid.uuid4().hex}{Path(safe_filename).suffix.lower()}"
    stored_path = RAW_DOCUMENTS_DIR / stored_name
    stored_path.write_bytes(contents)

    chunked = chunk_document(
        processed.text,
        document_id=processed.document_id,
        filename=processed.filename,
        page_numbers=[page.page_number for page in processed.pages],
    )

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
