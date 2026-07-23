"""Provider registry.

``get_provider`` resolves a name to an implementation. Cloud providers are
imported lazily so their optional SDKs are only required when actually selected.
"""

from __future__ import annotations

from seclit.config import Settings, settings
from seclit.generate.providers.base import LLMProvider
from seclit.generate.providers.ollama_provider import OllamaProvider

__all__ = ["LLMProvider", "OllamaProvider", "get_provider"]


def get_provider(
    name: str | None = None,
    config: Settings | None = None,
    model: str | None = None,
) -> LLMProvider:
    config = config or settings
    name = (name or config.provider).lower()

    if name == "ollama":
        return OllamaProvider(config, model)

    if name == "anthropic":
        from seclit.generate.providers.cloud import AnthropicProvider

        return AnthropicProvider(config, model)

    if name == "openai":
        from seclit.generate.providers.cloud import OpenAIProvider

        return OpenAIProvider(config, model)

    if name == "gemini":
        from seclit.generate.providers.cloud import GeminiProvider

        return GeminiProvider(config, model)

    raise ValueError(
        f"Unknown provider '{name}'. Choose one of: ollama, anthropic, openai, gemini."
    )
