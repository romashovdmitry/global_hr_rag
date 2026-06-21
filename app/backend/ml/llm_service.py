"""DIP-compliant LLM service: abstract interface + provider implementations.

Dependency Inversion Principle (D in SOLID):
- High-level modules (LangGraph nodes) depend on ``AbstractLLMService`` — an abstraction.
- The low-level detail (``ChatGroq``, ``ChatOllama``, etc.) is encapsulated in a
  concrete subclass and wired once at application startup.
- To swap the LLM provider for tests or a production migration, call
  ``set_llm_service(MyNewService())`` — zero changes in any node.

Liskov Substitution Principle (L in SOLID):
  Any subclass of ``AbstractLLMService`` can be passed wherever the abstract class
  is expected.  Both ``GroqLLMService`` and ``OllamaLLMService`` satisfy LSP: they
  implement all abstract methods without narrowing signatures or raising new exceptions.

Provider selection (auto-detected at startup via ``get_llm_service``):
  - ``GROQ_API_KEY`` set → ``GroqLLMService`` (cloud, LPU-accelerated)
  - otherwise            → ``OllamaLLMService`` (local fallback)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from backend.core.config import settings

logger = logging.getLogger(__name__)


class AbstractLLMService(ABC):
    """Abstract base class defining the LLM access interface for pipeline nodes.

    Both heavy and light models are exposed as separate methods so nodes
    explicitly communicate their compute requirements.  Subclasses must return
    objects that support ``.ainvoke(messages)`` and ``.with_structured_output(schema)``
    (LangChain Runnable interface).
    """

    @abstractmethod
    def get_llm(self, **kwargs: Any) -> Any:
        """Return a heavy chat model suitable for complex reasoning tasks.

        Used by: ``intent_classifier``, ``relevance_grader``, ``generator``,
        ``groundedness_grader``, ``summarization``, ``metadata_extractor``.

        Args:
            **kwargs: Model-specific overrides (e.g. ``temperature=0``).

        Returns:
            A chat model object supporting ``.ainvoke()`` and
            ``.with_structured_output()``.
        """

    @abstractmethod
    def get_light_llm(self, **kwargs: Any) -> Any:
        """Return a light chat model for fast structured-extraction tasks.

        Used by: ``query_translator``.
        A smaller/faster model is sufficient — deep reasoning is not required.

        Args:
            **kwargs: Model-specific overrides.

        Returns:
            A chat model object supporting ``.ainvoke()`` and
            ``.with_structured_output()``.
        """


class GroqLLMService(AbstractLLMService):
    """Concrete ``AbstractLLMService`` backed by the Groq cloud API.

    Groq uses LPU (Language Processing Unit) hardware for significantly faster
    inference than local CPU/GPU.  Requires a valid ``GROQ_API_KEY``.

    Heavy model default: ``llama-3.3-70b-versatile`` (70B — much higher quality
    than the previous local 3B model at comparable or lower latency).
    Light model default: ``llama-3.1-8b-instant`` (8B — fast structured tasks).
    """

    def __init__(
        self,
        llm_model: str,
        light_llm_model: str,
        api_key: str,
    ) -> None:
        """Initialise with model names and Groq API key.

        Args:
            llm_model: Heavy model name (e.g. ``"llama-3.3-70b-versatile"``).
            light_llm_model: Light model name (e.g. ``"llama-3.1-8b-instant"``).
            api_key: Groq API key.
        """
        self._llm_model = llm_model
        self._light_llm_model = light_llm_model
        self._api_key = api_key

    def get_llm(self, **kwargs: Any) -> ChatGroq:
        """Return a ``ChatGroq`` instance using the heavy model.

        Args:
            **kwargs: Passed directly to ``ChatGroq`` (e.g. ``temperature=0``).

        Returns:
            ``ChatGroq`` instance.
        """
        return ChatGroq(model=self._llm_model, api_key=self._api_key, **kwargs)

    def get_light_llm(self, **kwargs: Any) -> ChatGroq:
        """Return a ``ChatGroq`` instance using the light model.

        Args:
            **kwargs: Passed directly to ``ChatGroq``.

        Returns:
            ``ChatGroq`` instance.
        """
        return ChatGroq(model=self._light_llm_model, api_key=self._api_key, **kwargs)


class OllamaLLMService(AbstractLLMService):
    """Concrete ``AbstractLLMService`` backed by a local Ollama server.

    Used as a fallback when ``GROQ_API_KEY`` is not set.  All inference runs
    locally — no internet connection or API key required.
    """

    def __init__(
        self,
        llm_model: str,
        light_llm_model: str,
        base_url: str,
    ) -> None:
        """Initialise with model names and Ollama base URL.

        Args:
            llm_model: Heavy model name (e.g. ``"llama3.2"``).
            light_llm_model: Light model name (e.g. ``"llama3.2:1b"``).
            base_url: Ollama HTTP base URL (e.g. ``"http://ollama:11434"``).
        """
        self._llm_model = llm_model
        self._light_llm_model = light_llm_model
        self._base_url = base_url

    def get_llm(self, **kwargs: Any) -> ChatOllama:
        """Return a ``ChatOllama`` instance using the heavy model.

        Args:
            **kwargs: Passed directly to ``ChatOllama`` (e.g. ``temperature=0``).

        Returns:
            ``ChatOllama`` instance.
        """
        return ChatOllama(model=self._llm_model, base_url=self._base_url, **kwargs)

    def get_light_llm(self, **kwargs: Any) -> ChatOllama:
        """Return a ``ChatOllama`` instance using the light model.

        Args:
            **kwargs: Passed directly to ``ChatOllama``.

        Returns:
            ``ChatOllama`` instance.
        """
        return ChatOllama(model=self._light_llm_model, base_url=self._base_url, **kwargs)


_llm_service: AbstractLLMService | None = None


def get_llm_service() -> AbstractLLMService:
    """Return the active LLM service singleton.

    Auto-selects the provider on first call:
    - ``GROQ_API_KEY`` set → ``GroqLLMService`` (cloud, LPU-accelerated)
    - otherwise            → ``OllamaLLMService`` (local fallback)

    In tests, call ``set_llm_service(stub)`` before the first node execution to
    inject a mock without touching any node code.

    Returns:
        The currently registered ``AbstractLLMService`` instance.
    """
    global _llm_service
    if _llm_service is None:
        if settings.groq_api_key:
            logger.info(
                "LLM provider: Groq (heavy=%s, light=%s)",
                settings.groq_llm_model,
                settings.groq_light_llm_model,
            )
            _llm_service = GroqLLMService(
                llm_model=settings.groq_llm_model,
                light_llm_model=settings.groq_light_llm_model,
                api_key=settings.groq_api_key,
            )
        else:
            logger.info(
                "LLM provider: Ollama (heavy=%s, light=%s) — set GROQ_API_KEY to use Groq.",
                settings.llm_model,
                settings.light_llm_model,
            )
            _llm_service = OllamaLLMService(
                llm_model=settings.llm_model,
                light_llm_model=settings.light_llm_model,
                base_url=settings.ollama_host,
            )
    return _llm_service


def set_llm_service(service: AbstractLLMService) -> None:
    """Replace the active singleton — used in tests to inject a stub or mock.

    Any ``AbstractLLMService`` subclass can be passed here.  LSP guarantees that
    any valid subclass works correctly wherever the abstract type is expected.

    Args:
        service: Concrete ``AbstractLLMService`` instance to activate.
    """
    global _llm_service
    _llm_service = service
