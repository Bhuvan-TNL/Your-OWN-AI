from __future__ import annotations

import pytest

from app.services.llm import LLMConfigurationError, LLMProviderError, LLMService, build_prompt


def test_build_prompt_formats_context() -> None:
    prompt = build_prompt(
        "What is normalization?",
        "Normalization reduces redundancy and improves database integrity.",
    )

    assert "What is normalization?" in prompt
    assert "Normalization reduces redundancy and improves database integrity." in prompt
    assert "Context:" in prompt
    assert "use only the provided context" in prompt.lower()
    assert "do not use outside knowledge" in prompt.lower()


def test_mock_provider_generates_answer() -> None:
    service = LLMService(provider="mock", model="mock-model")
    result = service.generate_answer(
        "What is normalization?",
        "Normalization reduces redundancy and improves database integrity.",
    )

    assert result.answer
    assert result.provider == "mock"
    assert result.model == "mock-model"
    assert result.generation_time_ms >= 0
    assert "reduces redundancy" in result.answer


def test_prompt_can_use_configured_grounding_instructions() -> None:
    service = LLMService(
        provider="mock",
        model="mock-model",
        system_prompt="Answer only from context and say when evidence is missing.",
    )

    prompt = service.build_prompt("What is normalization?", "Normalization reduces redundancy.")

    assert "Answer only from context" in prompt
    assert "Normalization reduces redundancy." in prompt
    assert "What is normalization?" in prompt


def test_empty_context_is_explicitly_marked() -> None:
    service = LLMService(provider="mock", model="mock-model")

    prompt = service.build_prompt("What is normalization?", "")

    assert "No relevant context was retrieved" in prompt


def test_empty_question_is_rejected() -> None:
    service = LLMService(provider="mock", model="mock-model")

    with pytest.raises(LLMConfigurationError):
        service.build_prompt("   ", "relevant context")


def test_invalid_hf_model_is_not_valid_for_generation() -> None:
    service = LLMService(provider="huggingface", model="")

    with pytest.raises(LLMConfigurationError):
        service.generate_answer("What is this?", "Just context.")


def test_hf_provider_failure_is_reported_cleanly(monkeypatch) -> None:
    class DummyResponse:
        status_code = 503
        text = "model unavailable"

    def fake_post(*args, **kwargs):
        return DummyResponse()

    monkeypatch.setattr("app.services.llm.httpx.Client.post", fake_post)
    service = LLMService(provider="huggingface", model="google/flan-t5-base")

    with pytest.raises(LLMProviderError, match="status 503"):
        service.generate_answer("What is normalization?", "Normalization reduces redundancy.")


def test_hf_generation_uses_configured_request_parameters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            return [{"generated_text": "Normalization reduces redundancy."}]

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return DummyResponse()

    monkeypatch.setattr("app.services.llm.httpx.Client.post", fake_post)
    service = LLMService(
        provider="huggingface",
        model="google/flan-t5-base",
        max_tokens=42,
        temperature=0.7,
    )

    result = service.generate_answer("What is normalization?", "Normalization reduces redundancy.")

    assert result.answer == "Normalization reduces redundancy."
    assert captured["json"]["parameters"] == {
        "max_new_tokens": 42,
        "temperature": 0.7,
        "return_full_text": False,
    }
