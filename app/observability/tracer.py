"""
Трейсинг запроса от начала до конца.

Идея: на каждый /query запрос собираем один объект QueryTrace,
который фиксирует ВСЕ этапы (детекция языка, retrieval, rerank,
generation) с их latency и промежуточными результатами.

Это то, что отличает "у нас есть RAG" от "у нас есть RAG, и мы можем
объяснить почему конкретный ответ получился именно таким".
В интервью на позицию с уклоном в production reliability
это ключевая демонстрация — junior обычно логирует только финальный ответ.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.observability.logger import get_logger

logger = get_logger("query_trace")


@dataclass
class RetrievedChunkTrace:
    chunk_index: int
    source_filename: str
    page_number: int
    similarity_score: float
    rerank_score: float | None = None
    text_preview: str = ""  # первые ~100 символов, для читаемости лога


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_raw: str = ""
    query_language_detected: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # заполняются по ходу обработки
    retrieval_latency_ms: float | None = None
    retrieved_chunks: list[RetrievedChunkTrace] = field(default_factory=list)
    rerank_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    generation_input_tokens: int | None = None
    generation_output_tokens: int | None = None
    final_answer: str = ""
    error: str | None = None

    _stage_start: float = field(default=0.0, repr=False)

    def start_stage(self) -> None:
        self._stage_start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._stage_start) * 1000

    def total_latency_ms(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds() * 1000

    def emit(self) -> None:
        """
        Финальный вызов — пишем весь трейс одной структурированной записью.
        Вызывать в конце обработки запроса (успех или ошибка).
        """
        logger.info(
            "query_trace_complete",
            extra={
                "trace_id": self.trace_id,
                "query_raw": self.query_raw,
                "query_language": self.query_language_detected,
                "retrieval_latency_ms": self.retrieval_latency_ms,
                "rerank_latency_ms": self.rerank_latency_ms,
                "generation_latency_ms": self.generation_latency_ms,
                "total_latency_ms": round(self.total_latency_ms(), 1),
                "generation_input_tokens": self.generation_input_tokens,
                "generation_output_tokens": self.generation_output_tokens,
                "num_chunks_retrieved": len(self.retrieved_chunks),
                "top_chunk_sources": [
                    f"{c.source_filename}#p{c.page_number}" for c in self.retrieved_chunks[:3]
                ],
                "error": self.error,
            },
        )
