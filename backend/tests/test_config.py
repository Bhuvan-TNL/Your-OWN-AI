from __future__ import annotations

import os

import pytest

from app.core.config import _load_dotenv


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_PROVIDER", "TOP_K"):
        monkeypatch.delenv(key, raising=False)


def test_load_dotenv_reads_plain_utf8(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=huggingface\nTOP_K=5\n",
        encoding="utf-8",
    )

    _load_dotenv(env_file)

    assert os.environ["LLM_PROVIDER"] == "huggingface"
    assert os.environ["TOP_K"] == "5"


def test_load_dotenv_strips_utf8_bom(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=huggingface\nTOP_K=5\n",
        encoding="utf-8-sig",
    )

    _load_dotenv(env_file)

    assert os.environ["LLM_PROVIDER"] == "huggingface"
    assert os.environ["TOP_K"] == "5"


def test_load_dotenv_missing_file_is_a_noop(tmp_path):
    missing_path = tmp_path / "does-not-exist.env"

    _load_dotenv(missing_path)

    assert "LLM_PROVIDER" not in os.environ