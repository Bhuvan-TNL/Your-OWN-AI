import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.chunker import chunk_text, validate_chunk_parameters
from app.services.doc_processor import EmptyDocumentError, UnsupportedFileTypeError, process_document


client = TestClient(app)


def test_txt_extraction_and_metadata() -> None:
    document = process_document(
        b"First line of content.\n\nSecond line of content.",
        filename="notes.txt",
        document_id="doc-001",
    )

    assert document.file_type == "txt"
    assert document.filename == "notes.txt"
    assert document.document_id == "doc-001"
    assert "First line of content" in document.text
    assert document.pages[0].page_number == 1


def test_chunking_generates_metadata() -> None:
    chunks = chunk_text(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        document_id="doc-002",
        filename="notes.txt",
        chunk_size=5,
        chunk_overlap=2,
        page_number=1,
    )

    assert len(chunks) >= 1
    assert chunks[0].document_id == "doc-002"
    assert chunks[0].filename == "notes.txt"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id.startswith("doc-002-chunk-")
    assert chunks[0].text


def test_invalid_chunk_parameters() -> None:
    with pytest.raises(ValueError):
        validate_chunk_parameters(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        validate_chunk_parameters(chunk_size=5, chunk_overlap=5)


def test_unsupported_file_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        process_document(b"This is not a valid file type.", filename="sample.docx")


def test_empty_document() -> None:
    with pytest.raises(EmptyDocumentError):
        process_document(b"   \n\t  ", filename="empty.txt")


def test_upload_endpoint_validates_and_processes_txt() -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("sample.txt", b"Alpha beta gamma delta epsilon zeta eta.\n\nMore content after the break.", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_type"] == "txt"
    assert payload["page_count"] == 1
    assert payload["chunk_count"] >= 1
    assert payload["stored_path"]
    assert payload["chunks"][0]["chunk_id"]
