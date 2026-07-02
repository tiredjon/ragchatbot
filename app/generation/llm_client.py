"""
Обёртка над Gemini API. Один клиент на два разных use case:
1. Эмбеддинги (для ingestion и retrieval)
2. Генерация ответа (для финального ответа пользователю)

Почему обёртка, а не прямые вызовы SDK по всему коду:
- Единая точка для retry/error handling
- Единая точка для логирования latency и token usage (нужно для observability)
- Если завтра решишь сменить провайдера — меняешь один файл, а не 10
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import google.generativeai as genai

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    model: str


@dataclass
class EmbeddingResult:
    vector: list[float]
    latency_ms: float
    model: str


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self._generation_model_name = settings.gemini_generation_model
        self._embedding_model_name = settings.gemini_embedding_model
        self._generation_model = genai.GenerativeModel(self._generation_model_name)

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> EmbeddingResult:
        """
        task_type различается для документов (то что кладём в базу)
        и запросов (то что ищем) — Gemini использует это для оптимизации
        эмбеддинга под задачу. Используй 'retrieval_query' при поиске.
        """
        start = time.perf_counter()
        response = genai.embed_content(
            model=f"models/{self._embedding_model_name}",
            content=text,
            task_type=task_type,
            output_dimensionality=768,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            vector=response["embedding"],
            latency_ms=latency_ms,
            model=self._embedding_model_name,
        )

    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[EmbeddingResult]:
        """
        Батчевый эмбеддинг для ingestion — без этого загрузка большого
        документа будет делать сотни последовательных API-вызовов.
        """
        results = []
        for text in texts:
            # Gemini free tier API пока не поддерживает батч-эмбеддинги в одном
            # вызове стабильно для всех SDK версий, поэтому идём по одному,
            # но с общим client instance (переиспользуем connection).
            results.append(self.embed_text(text, task_type=task_type))
        return results

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.4,
        max_output_tokens: int = 5000,
    ) -> GenerationResult:
        """
        temperature=0.4 по умолчанию — баланс между фактологичностью
        (нужной для RAG) и достаточной свободой, чтобы модель не сжимала
        ответ до одной сухой фразы. max_output_tokens=2048 даёт запас
        на развёрнутый ответ с контекстом, а не только короткий факт.
        """
        start = time.perf_counter()

        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = self._generation_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Gemini SDK отдаёт usage_metadata не всегда одинаково между версиями,
        # поэтому достаём осторожно
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

        logger.info(
            "llm_generation",
            extra={
                "model": self._generation_model_name,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

        return GenerationResult(
            text=response.text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._generation_model_name,
        )


_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Singleton — не пересоздаём клиент на каждый запрос."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance
