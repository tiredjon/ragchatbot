"""
Retrieval: векторный поиск в pgvector + reranking.

Почему нужен rerank, а не просто топ-K по косинусному сходству:
Vector similarity search быстрый, но грубый — он находит "похожие по смыслу"
куски, но не всегда самые РЕЛЕВАНТНЫЕ конкретному вопросу. Rerank —
это второй, более дорогой проход, который пересматривает top_k_retrieval
кандидатов и выбирает top_k_after_rerank лучших через LLM-scoring.

Это именно то отличие "просто similarity search" от "production RAG",
про которое я говорил как про ключевой гэп для senior-позиционирования.
"""
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.generation.llm_client import get_llm_client
from app.observability.tracer import QueryTrace, RetrievedChunkTrace


class RetrievedChunk:
    def __init__(
        self,
        chunk_index: int,
        source_filename: str,
        page_number: int,
        text: str,
        similarity_score: float,
    ) -> None:
        self.chunk_index = chunk_index
        self.source_filename = source_filename
        self.page_number = page_number
        self.text = text
        self.similarity_score = similarity_score
        self.rerank_score: float | None = None


def vector_search(
    query_embedding: list[float],
    document_id: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Косинусный поиск в pgvector, отфильтрованный по document_id
    (пользователь ищет только в рамках своего загруженного документа).

    Оператор <=> в pgvector — это cosine distance, поэтому сортируем
    по возрастанию (меньше расстояние = более похоже), а для similarity_score
    конвертируем в привычный 0..1, где 1 = полное совпадение.
    """
    settings = get_settings()

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    chunk_index, source_filename, page_number, text,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, document_id, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_index=row["chunk_index"],
            source_filename=row["source_filename"],
            page_number=row["page_number"],
            text=row["text"],
            similarity_score=row["similarity"],
        )
        for row in rows
    ]


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int,
) -> list[RetrievedChunk]:
    """
    LLM-based rerank: просим Gemini оценить релевантность каждого чанка
    вопросу по шкале 0-10, затем берём top_n.

    Это медленнее чем cross-encoder rerank модель, но не требует
    дополнительной зависимости/инфраструктуры — приемлемый trade-off
    для портфолио-проекта. В README стоит явно отметить это ограничение
    и написать, чем заменить в реальном проде (например bge-reranker).
    """
    if not chunks:
        return []

    client = get_llm_client()

    scoring_prompt = _build_rerank_prompt(query, chunks)
    result = client.generate(
        system_prompt=(
            "Ты оцениваешь релевантность фрагментов текста вопросу пользователя. "
            "Отвечай СТРОГО в формате: одна строка на фрагмент, 'INDEX: SCORE', "
            "где SCORE — число от 0 до 10. Без пояснений."
        ),
        user_message=scoring_prompt,
        temperature=0.0,
    )

    scores = _parse_rerank_scores(result.text, num_chunks=len(chunks))

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = score

    ranked = sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)
    return ranked[:top_n]


def _build_rerank_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines = [f"Вопрос: {query}\n", "Фрагменты:"]
    for i, chunk in enumerate(chunks):
        preview = chunk.text[:400]
        lines.append(f"[{i}] {preview}")
    return "\n\n".join(lines)


def _parse_rerank_scores(response_text: str, num_chunks: int) -> list[float]:
    """
    Парсит ответ модели вида '0: 8\n1: 3\n2: 9'.
    При ошибке парсинга — fallback на нейтральный скор 5.0 для всех,
    чтобы rerank не уронил весь запрос, а просто не улучшил порядок.
    """
    scores = [5.0] * num_chunks
    for line in response_text.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        try:
            idx_str, score_str = line.split(":", 1)
            idx = int(idx_str.strip().strip("[]"))
            score = float(score_str.strip())
            if 0 <= idx < num_chunks:
                scores[idx] = score
        except (ValueError, IndexError):
            continue
    return scores


def retrieve(query: str, document_id: str, trace: QueryTrace) -> list[RetrievedChunk]:
    """
    Полный retrieval pipeline: embed query -> vector search -> rerank.
    Пишет все промежуточные результаты в trace для observability.
    """
    settings = get_settings()
    client = get_llm_client()

    trace.start_stage()
    query_embedding = client.embed_text(query, task_type="retrieval_query").vector
    candidates = vector_search(
        query_embedding=query_embedding,
        document_id=document_id,
        top_k=settings.top_k_retrieval,
    )
    trace.retrieval_latency_ms = trace.elapsed_ms()

    trace.start_stage()
    top_chunks = rerank_chunks(query, candidates, top_n=settings.top_k_after_rerank)
    trace.rerank_latency_ms = trace.elapsed_ms()

    trace.retrieved_chunks = [
        RetrievedChunkTrace(
            chunk_index=c.chunk_index,
            source_filename=c.source_filename,
            page_number=c.page_number,
            similarity_score=round(c.similarity_score, 4),
            rerank_score=c.rerank_score,
            text_preview=c.text[:120],
        )
        for c in top_chunks
    ]

    return top_chunks
