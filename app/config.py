"""
Централизованная конфигурация приложения.
Все настройки читаются из переменных окружения (.env),
чтобы не хардкодить ключи и не гонять секреты в git.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Gemini API ---
    gemini_api_key: str
    gemini_generation_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # --- Supabase / Postgres (pgvector) ---
    supabase_url: str
    supabase_service_key: str
    database_url: str  # прямой postgres connection string для pgvector запросов

    # --- Retrieval tuning ---
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75
    top_k_retrieval: int = 8
    top_k_after_rerank: int = 4

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_mb: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """
    Кэшируем настройки, чтобы не парсить .env на каждый запрос.
    Использование: settings = get_settings()
    """
    return Settings()
