"""Ollama provider — the default, fully local path."""

from __future__ import annotations

from collections.abc import Iterator

from seclit.config import Settings, settings
from seclit.generate.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config: Settings | None = None, model: str | None = None) -> None:
        self.config = config or settings
        self.model = model or self.config.ollama_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self.config.ollama_host)
        return self._client

    def available(self) -> tuple[bool, str]:
        """Distinguish "daemon down" from "model not pulled" — different fixes."""
        try:
            response = self.client.list()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
            return False, (
                f"Cannot reach Ollama at {self.config.ollama_host} ({exc}). "
                "Start it with `ollama serve`."
            )

        available = {m.get("model") or m.get("name", "") for m in (response.get("models") or [])}
        if not any(m == self.model or m.startswith(f"{self.model}:") for m in available):
            return False, (
                f"Model '{self.model}' is not installed. Pull it with `ollama pull {self.model}`."
            )
        return True, "ok"

    def _options(self, **kwargs) -> dict:
        return {
            "num_predict": kwargs.get("max_tokens", self.config.max_output_tokens),
            # Low but non-zero: grounded synthesis benefits from a little
            # flexibility in phrasing, while high temperature invites the model
            # to embellish beyond the retrieved context.
            "temperature": kwargs.get("temperature", 0.2),
        }

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        response = self.client.chat(
            model=self.model,
            messages=self._messages(prompt, system),
            options=self._options(**kwargs),
        )
        return response["message"]["content"]

    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        for part in self.client.chat(
            model=self.model,
            messages=self._messages(prompt, system),
            options=self._options(**kwargs),
            stream=True,
        ):
            piece = part.get("message", {}).get("content", "")
            if piece:
                yield piece
