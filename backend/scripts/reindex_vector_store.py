"""Safely rebuild the generated FAISS index from local raw documents.

The command is intentionally dry-run by default. Use ``--apply`` to replace
only ``faiss_index.bin`` and ``metadata.json`` under the configured vector
store directory. Raw documents are never removed or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.core.config import BACKEND_DIR, settings
from app.services.chunker import chunk_document
from app.services.doc_processor import process_document
from app.services.embeddings import embed_texts
from app.services.vector_store import VectorStore

RAW_DOCUMENTS_DIR = BACKEND_DIR / "data" / "raw_documents"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
UUID_SUFFIX = re.compile(r"-[0-9a-f]{32}(?=\.[^.]+$)", re.IGNORECASE)


def _logical_filename(path: Path) -> str:
    """Recover the original upload filename from the stored random suffix."""

    return UUID_SUFFIX.sub("", path.name)


def _content_hash(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _source_documents(source_dir: Path) -> list[tuple[Path, bytes, str]]:
    if not source_dir.exists():
        return []

    documents: list[tuple[Path, bytes, str]] = []
    seen_hashes: set[str] = set()
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        contents = path.read_bytes()
        digest = _content_hash(contents)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        documents.append((path, contents, digest))
    return documents


def _processed_chunks(source_dir: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    all_chunks: list[Any] = []
    documents: list[dict[str, Any]] = []
    for path, contents, digest in _source_documents(source_dir):
        filename = _logical_filename(path)
        document_id = f"doc-{digest[:16]}"
        processed = process_document(contents, filename=filename, document_id=document_id)
        chunks = chunk_document(
            processed.text,
            document_id=processed.document_id,
            filename=processed.filename,
            page_content=[(page.page_number, page.text) for page in processed.pages],
            content_hash=digest,
        )
        all_chunks.extend(chunks)
        documents.append(
            {
                "document_id": document_id,
                "filename": filename,
                "content_hash": digest,
                "page_count": len(processed.pages),
                "chunk_count": len(chunks),
            }
        )
    return all_chunks, documents


def _safe_clear_generated_index(index_dir: Path) -> None:
    resolved = index_dir.resolve()
    if resolved in {BACKEND_DIR.resolve(), BACKEND_DIR.parent.resolve()} or resolved.name.lower() not in {
        "vector_store",
        "index",
    }:
        raise ValueError(f"Refusing to clear unexpected vector-store directory: {resolved}")

    for filename in ("faiss_index.bin", "metadata.json"):
        target = resolved / filename
        if target.exists():
            target.unlink()


def rebuild(source_dir: Path, *, apply: bool) -> dict[str, Any]:
    chunks, documents = _processed_chunks(source_dir)
    index_dir = Path(settings.faiss_index_path)
    result: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "source_dir": str(source_dir.resolve()),
        "index_dir": str(index_dir.resolve()),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents": documents,
    }

    if not apply:
        return result

    _safe_clear_generated_index(index_dir)
    store = VectorStore(index_path=index_dir, top_k=settings.top_k)
    vectors = embed_texts([chunk.text for chunk in chunks])
    store.add_chunks(chunks, vectors)
    store.save()
    result["metadata_count"] = len(store.metadata)
    result["unique_content_hashes"] = len(
        {metadata.get("content_hash") for metadata in store.metadata if metadata.get("content_hash")}
    )
    result["page_numbers"] = sorted(
        {metadata.get("page_number") for metadata in store.metadata if metadata.get("page_number") is not None}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_DOCUMENTS_DIR,
        help="Directory containing raw PDF/TXT documents (default: backend/data/raw_documents).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace only generated FAISS/metadata files. Without this flag, perform a dry run.",
    )
    args = parser.parse_args()
    print(json.dumps(rebuild(args.source_dir, apply=args.apply), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
