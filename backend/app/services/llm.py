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
    def generate(self, prompt: str) -> str:
        ...


class MockLLMProvider:
    """Local fallback used for tests and safe local development without external APIs."""

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        cleaned = prompt.strip()
        if not cleaned:
            raise LLMConfigurationError("Prompt cannot be empty.")
        if "context:" in cleaned.lower():
            return "Based on the provided context, the answer is grounded in the retrieved material."
        return "This is a mock LLM response generated locally for development and testing."


class HuggingFaceLLMProvider:
    """Provider-neutral wrapper for Hugging Face inference calls."""

    def __init__(self, model: str, token: str | None = None) -> None:
        self.model = model
        self.token = token or ""

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise LLMConfigurationError("Prompt cannot be empty.")
        if not self.model or self.model == "your_model_here":
            raise LLMConfigurationError(
                "LLM_MODEL is not configured. Set it in backend/.env or environment variables."
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.2,
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
            generated_text = data[0].get("generated_text") if isinstance(data[0], dict) else str(data[0])
        elif isinstance(data, dict):
            generated_text = data.get("generated_text")
            if generated_text is None and "error" in data:
                raise LLMProviderError(str(data["error"]))
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
    ) -> None:
        self.provider = (provider or settings.llm_provider or "huggingface").strip().lower()
        self.model = model or settings.llm_model or "your_model_here"
        self.token = token if token is not None else settings.hf_token

        if self.provider == "mock":
            self._provider: BaseLLMProvider = MockLLMProvider(model=self.model)
        elif self.provider == "huggingface":
            self._provider = HuggingFaceLLMProvider(model=self.model, token=self.token)
        else:
            raise LLMConfigurationError(f"Unsupported LLM provider '{self.provider}'.")

    def build_prompt(self, question: str, context: str) -> str:
        clean_question = (question or "").strip()
        clean_context = (context or "").strip()

        if not clean_question:
            raise LLMConfigurationError("Question cannot be empty.")

        if not clean_context:
            return (
                "You are a helpful assistant. Answer the user's question using only the available knowledge. "
                f"Question: {clean_question}"
            )

        return (
            "You are a grounded assistant. Use the provided context to answer the user's question. "
            "If the context does not contain enough information, say so clearly.\n\n"
            f"Context:\n{clean_context}\n\nQuestion:\n{clean_question}"
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
            answer = self._provider.generate(prompt)
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
    service = LLMService(provider=provider or settings.llm_provider, model=model or settings.llm_model, token=settings.hf_token)
    return service.generate_answer(question, context)
