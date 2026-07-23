"""Cloud provider adapters: Anthropic, OpenAI, Gemini.

None of these are required to run the application — the default path is local.
They exist so the same retrieval and citation machinery can be pointed at a
hosted model by changing ``SECLIT_PROVIDER``, and so the recommendation in the
project write-up is backed by working code rather than an assertion.

Each SDK is imported inside its provider, so a missing optional dependency
surfaces as a clear message from ``available()`` rather than an ImportError at
startup.
"""

from __future__ import annotations

from collections.abc import Iterator

from seclit.config import Settings, settings
from seclit.generate.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Claude. The recommended hosted option for this workload: long context
    for many excerpts at once, and reliable adherence to the marker format that
    citation validation depends on."""

    name = "anthropic"

    def __init__(self, config: Settings | None = None, model: str | None = None) -> None:
        self.config = config or settings
        self.model = model or self.config.anthropic_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "Install with: uv sync --extra anthropic"
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set."
        return True, "ok"

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: Settings | None = None, model: str | None = None) -> None:
        self.config = config or settings
        self.model = model or self.config.openai_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def available(self) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "Install with: uv sync --extra openai"
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set."
        return True, "ok"

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            max_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            temperature=kwargs.get("temperature", 0.2),
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            max_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            temperature=kwargs.get("temperature", 0.2),
            stream=True,
        ):
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: Settings | None = None, model: str | None = None) -> None:
        self.config = config or settings
        self.model = model or self.config.gemini_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def available(self) -> tuple[bool, str]:
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False, "Install with: uv sync --extra gemini"
        import os

        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            return False, "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set."
        return True, "ok"

    def _config(self, system: str | None, **kwargs):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=kwargs.get("max_tokens", self.config.max_output_tokens),
            temperature=kwargs.get("temperature", 0.2),
        )

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system, **kwargs)
        )
        return response.text or ""

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=prompt, config=self._config(system, **kwargs)
        ):
            if chunk.text:
                yield chunk.text
