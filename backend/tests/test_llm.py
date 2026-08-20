from __future__ import annotations

import pytest

from app.services.llm import LLMConfigurationError, LLMService, build_prompt


def test_build_prompt_formats_context() -> None:
    prompt = build_prompt(
        "What is normalization?",
        "Normalization reduces redundancy and improves database integrity.",
    )

    assert "What is normalization?" in prompt
    assert "Normalization reduces redundancy and improves database integrity." in prompt
    assert "Context:" in prompt


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


def test_empty_question_is_rejected() -> None:
    service = LLMService(provider="mock", model="mock-model")

    with pytest.raises(LLMConfigurationError):
        service.build_prompt("   ", "relevant context")


def test_placeholder_hf_model_is_not_valid_for_generation() -> None:
    service = LLMService(provider="huggingface", model="your_model_here")

    with pytest.raises(LLMConfigurationError):
        service.generate_answer("What is this?", "Just context.")
