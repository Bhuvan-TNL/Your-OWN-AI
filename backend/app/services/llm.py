from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import settings

logger = logging.getLogger("your_own_ai.llm")


class LLMConfigurationError(ValueError):
    """Raised when the configured LLM provider or model is invalid."""


class LLMProviderError(RuntimeError):
    """Raised when a provider call fails."""


@dataclass(frozen=True)
class LLMGenerationResult:
    answer: str
    provider: str
    model: str
    generation_time_ms: float
    tokens_used: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "generation_time_ms": self.generation_time_ms,
            "tokens_used": self.tokens_used,
        }


class BaseLLMProvider(Protocol):
    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        ...


class MockLLMProvider:
    """Local fallback used for tests and safe local development without external APIs."""

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        cleaned = prompt.strip()
        if not cleaned:
            raise LLMConfigurationError("Prompt cannot be empty.")
        context_marker = "\ncontext:\n"
        question_marker = "\n\nquestion:"
        context_start = cleaned.lower().find(context_marker)
        if context_start >= 0:
            context = cleaned[context_start + len(context_marker) :]
            question_start = context.lower().find(question_marker)
            if question_start >= 0:
                context = context[:question_start]
            context_lines = [line.strip() for line in context.splitlines() if line.strip()]
            source_lines = [line for line in context_lines if line.startswith("[Source:")]
            answer_lines = [line for line in context_lines if not line.startswith("[Source:")]
            if answer_lines:
                excerpt = answer_lines[0]
                return f"Based on the provided context: {excerpt}"
            if source_lines:
                return "Based on the provided context, the retrieved source is relevant to the question."
        return "This is a mock LLM response generated locally for development and testing."


class HuggingFaceLLMProvider:
    """Provider-neutral wrapper for Hugging Face inference calls."""

    def __init__(
        self,
        model: str,
        token: str | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.token = token or ""
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        if not prompt or not prompt.strip():
            raise LLMConfigurationError("Prompt cannot be empty.")
        if not self.model or self.model.strip() in {"", "your_model_here"}:
            raise LLMConfigurationError(
                "LLM_MODEL is not configured. Set it in backend/.env or environment variables."
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            },
        }

        url = f"https://api-inference.huggingface.co/models/{self.model}"
        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"Hugging Face request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text.strip() or "unknown error"
            raise LLMProviderError(f"Hugging Face request failed with status {response.status_code}: {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError("Hugging Face returned invalid JSON.") from exc

        if isinstance(data, list):
            if not data:
                raise LLMProviderError("Hugging Face returned an empty response.")
            first_item = data[0]
            generated_text = first_item.get("generated_text") if isinstance(first_item, dict) else str(first_item)
        elif isinstance(data, dict):
            if "error" in data:
                raise LLMProviderError(str(data["error"]))
            generated_text = data.get("generated_text")
        else:
            generated_text = str(data)

        if not generated_text:
            raise LLMProviderError("Hugging Face returned no generated text.")

        return str(generated_text).strip()


class LLMService:
    """Service for generating answers from a question and retrieved context."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        token: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        raw_provider = provider if provider is not None else settings.llm_provider
        self.provider = (raw_provider or "huggingface").strip().lower()
        raw_model = model if model is not None else settings.llm_model
        self.model = raw_model.strip() if isinstance(raw_model, str) else (raw_model or "google/flan-t5-base")
        self.token = token if token is not None else settings.hf_token
        self.system_prompt = (system_prompt if system_prompt is not None else settings.llm_system_prompt).strip()
        self.max_tokens = int(max_tokens if max_tokens is not None else settings.llm_max_tokens)
        self.temperature = float(temperature if temperature is not None else settings.llm_temperature)

        if not self.system_prompt:
            raise LLMConfigurationError("LLM_SYSTEM_PROMPT cannot be empty.")
        if self.max_tokens <= 0:
            raise LLMConfigurationError("LLM_MAX_TOKENS must be greater than 0.")
        if not 0.0 <= self.temperature <= 2.0:
            raise LLMConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")

        if self.provider == "mock":
            self._provider: BaseLLMProvider = MockLLMProvider(model=self.model)
        elif self.provider == "huggingface":
            self._provider = HuggingFaceLLMProvider(
                model=self.model,
                token=self.token,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        else:
            raise LLMConfigurationError(f"Unsupported LLM provider '{self.provider}'.")

    def build_prompt(self, question: str, context: str) -> str:
        clean_question = (question or "").strip()
        clean_context = (context or "").strip()

        if not clean_question:
            raise LLMConfigurationError("Question cannot be empty.")

        context_section = clean_context or "(No relevant context was retrieved.)"
        return (
            f"System instructions:\n{self.system_prompt}\n\n"
            "Context:\n"
            f"{context_section}\n\n"
            f"Question:\n{clean_question}\n\n"
            "Answer using only the context above. If it is insufficient, say so clearly."
        )

    def generate_answer(
        self,
        question: str,
        context: str,
        *,
        max_tokens: int | None = None,
    ) -> LLMGenerationResult:
        prompt = self.build_prompt(question, context)
        started_at = time.perf_counter()

        try:
            if max_tokens is None:
                answer = self._provider.generate(prompt)
            else:
                answer = self._provider.generate(prompt, max_tokens=max_tokens)
        except LLMConfigurationError:
            raise
        except LLMProviderError as exc:
            logger.exception("LLM generation failed for provider=%s model=%s", self.provider, self.model)
            raise LLMProviderError(f"LLM generation failed: {exc}") from exc

        generation_time_ms = (time.perf_counter() - started_at) * 1000
        return LLMGenerationResult(
            answer=answer,
            provider=self.provider,
            model=self.model,
            generation_time_ms=generation_time_ms,
            tokens_used=max_tokens,
        )


llm_service = LLMService(provider=settings.llm_provider, model=settings.llm_model, token=settings.hf_token)


def build_prompt(question: str, context: str) -> str:
    return llm_service.build_prompt(question, context)


def generate_answer(question: str, context: str, *, provider: str | None = None, model: str | None = None) -> LLMGenerationResult:
    service = LLMService(
        provider=provider if provider is not None else settings.llm_provider,
        model=model if model is not None else settings.llm_model,
        token=settings.hf_token,
    )
    return service.generate_answer(question, context)
