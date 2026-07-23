"""Provider interface.

Generation sits behind a two-method interface so the LLM is a configuration
choice rather than an architectural commitment. The default path is a local
Ollama model — no API key, no per-query cost, and the "local execution"
requirement satisfied literally. Switching to a hosted model is one environment
variable, with no change to retrieval, prompting, or citation validation.

Keeping the surface this small is deliberate: every provider must support
exactly streaming and non-streaming completion, so none of them can leak
provider-specific behaviour into the rest of the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    """Minimal text-generation interface."""

    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Return a full completion."""

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None, **kwargs) -> Iterator[str]:
        """Yield completion text incrementally."""

    def available(self) -> tuple[bool, str]:
        """Report whether the provider is usable right now.

        Returns ``(ok, detail)``. Used by the UI to fail with an actionable
        message ("Ollama is not running") instead of a stack trace mid-answer.
        """
        return True, "ok"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.model!r}>"
