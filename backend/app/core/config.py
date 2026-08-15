from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BACKEND_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DOTENV_PATH: Final[Path] = BACKEND_DIR / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def _as_positive_int(value: str, *, field_name: str, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive validation
        raise ValueError(f"{field_name} must be an integer.") from exc

    if parsed < minimum:
        raise ValueError(f"{field_name} must be greater than or equal to {minimum}.")

    return parsed


def _as_non_negative_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive validation
        raise ValueError(f"{field_name} must be an integer.") from exc

    if parsed < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")

    return parsed


def _resolve_output_path(raw_value: str) -> str:
    configured_path = Path(raw_value)
    if configured_path.is_absolute():
        return str(configured_path)
    return str((BACKEND_DIR / configured_path).resolve())


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    embedding_model: str
    faiss_index_path: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hf_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(DOTENV_PATH)

        llm_provider = (os.getenv("LLM_PROVIDER", "huggingface") or "huggingface").strip().lower()
        if not llm_provider:
            raise ValueError("LLM_PROVIDER cannot be empty.")

        llm_model = (os.getenv("LLM_MODEL", "your_model_here") or "your_model_here").strip() or "your_model_here"

        embedding_model = (os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2"

        faiss_index_path = _resolve_output_path(
            (os.getenv("FAISS_INDEX_PATH", "./data/vector_store") or "./data/vector_store").strip()
        )

        chunk_size = _as_positive_int(os.getenv("CHUNK_SIZE", "512"), field_name="CHUNK_SIZE")
        chunk_overlap = _as_non_negative_int(os.getenv("CHUNK_OVERLAP", "50"), field_name="CHUNK_OVERLAP")
        if chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        top_k = _as_positive_int(os.getenv("TOP_K", "5"), field_name="TOP_K")
        hf_token = (os.getenv("HF_TOKEN", "") or "").strip()

        return cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            faiss_index_path=faiss_index_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            hf_token=hf_token,
        )


settings = Settings.from_env()
