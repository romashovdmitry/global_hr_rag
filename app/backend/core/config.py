"""Global application configuration loaded from environment variables."""

from __future__ import annotations

import json

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised settings for the RAG backend.

    All values are read from environment variables (or .env file).

    LLM provider selection (auto-detected at startup):
    - If GROQ_API_KEY is set → GroqLLMService (cloud, fast LPU inference)
    - Otherwise              → OllamaLLMService (local, no API key required)
    """

    # PostgreSQL
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "rags"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "vacancies"

    # Groq cloud inference (takes priority over Ollama when set)
    groq_api_key: str = ""
    # Heavy model: generation, relevance grading, groundedness, intent classification
    groq_llm_model: str = "llama-3.3-70b-versatile"
    # Light model: query translation and other low-risk structured tasks
    groq_light_llm_model: str = "llama-3.1-8b-instant"

    # Ollama (local fallback when GROQ_API_KEY is not set)
    ollama_host: str = "http://ollama:11434"
    llm_model: str = "llama3.2"
    light_llm_model: str = "llama3.2:1b"

    # CORS — comma-separated list of allowed origins
    backend_cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        value = self.backend_cors_origins.strip()
        if value.startswith("["):
            return json.loads(value)
        return [o.strip() for o in value.split(",") if o.strip()]

    # Paths
    vacancies_json_path: str = "/app/vacancies.json"

    # RAG — dense embeddings via fastembed (local, no API key)
    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    dense_vector_size: int = 384
    sparse_embedding_model: str = "Qdrant/bm25"
    rag_max_retries: int = 3
    retrieval_top_k: int = 10
    rrf_k: int = 60

    @property
    def database_url(self) -> str:
        """Async SQLAlchemy connection string for PostgreSQL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        """Full Qdrant HTTP URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()

# Fixed conversation ID for single-user mode.
# The backend always reads/writes to this one row in chat_sessions.
CONVERSATION_ID = "00000000-0000-0000-0000-000000000001"
