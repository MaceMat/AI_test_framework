"""LLM client abstraction used by the judge and semantic checks.

The framework never hard-codes a provider. Pick one with environment vars:

    AITEST_LLM_PROVIDER = fake | openai | azure      (default: fake)
    AITEST_JUDGE_MODEL  = model / Azure deployment name for the judge
    AITEST_EMBED_MODEL  = embedding model / deployment (optional)

    # openai
    OPENAI_API_KEY
    # azure
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION

``fake`` keeps unit tests and pipeline smoke runs free and deterministic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings, or None if this client has no embedding model."""
        ...


class FakeLLM:
    """Deterministic stand-in for an LLM.

    ``rules`` maps a regex (searched in the prompt) to a response string, or is
    a callable(prompt) -> str. Unmatched prompts get ``default``. Every call is
    stored in ``calls`` so tests can assert on what was sent.

    With no rules it behaves as a lenient judge: it returns a passing JSON verdict.
    """

    def __init__(
        self,
        rules: dict[str, str] | Callable[[str], str] | None = None,
        default: str | None = None,
    ) -> None:
        self.rules = rules or {}
        self.default = default if default is not None else json.dumps({"score": 1.0, "reason": "fake judge"})
        self.calls: list[dict[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if callable(self.rules):
            return self.rules(prompt)
        for pattern, response in self.rules.items():
            if re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL):
                return response
        return self.default

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        return None  # semantic checks fall back to lexical similarity


class OpenAIClient:
    """OpenAI or Azure OpenAI via the official SDK (``pip install aitest[openai]``)."""

    def __init__(self, azure: bool = False, model: str | None = None, embed_model: str | None = None) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the OpenAI SDK: pip install 'aitest[openai]'") from e

        if azure:
            self._client = openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        else:
            self._client = openai.OpenAI()
        self.model = model or os.environ.get("AITEST_JUDGE_MODEL", "gpt-4o-mini")
        self.embed_model = embed_model or os.environ.get("AITEST_EMBED_MODEL")

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.embed_model:
            return None
        resp = self._client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]


def get_llm(provider: str | None = None) -> LLMClient:
    provider = (provider or os.environ.get("AITEST_LLM_PROVIDER", "fake")).lower()
    if provider == "fake":
        return FakeLLM()
    if provider == "openai":
        return OpenAIClient(azure=False)
    if provider == "azure":
        return OpenAIClient(azure=True)
    raise ValueError(f"Unknown AITEST_LLM_PROVIDER '{provider}' (use fake | openai | azure)")
