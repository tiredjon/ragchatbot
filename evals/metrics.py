"""
Метрики для оценки качества RAG-агента.

Три метрики, которые реально показывают производственную надёжность
(а не просто "модель что-то ответила"):

1. Retrieval Precision@K — нашёл ли retriever правильную страницу-источник
   в топ-K кандидатах. Если эта метрика низкая — проблема в чанкинге
   или эмбеддингах, а не в генерации.

2. Answer Relevance — использует LLM-as-judge: отдельный вызов модели
   оценивает, отвечает ли сгенерированный текст на заданный вопрос
   (независимо от фактической точности).

3. Groundedness (защита от галлюцинаций) — LLM-as-judge проверяет,
   опирается ли ответ ТОЛЬКО на предоставленный контекст, или модель
   что-то "довыдумала" от себя. Это самая важная метрика для продакшена,
   именно она отличает надёжный RAG от "обычно работает".

Плюс простая эвристика "refusal correctness" — правильно ли агент
отказался отвечать на вопрос не по документу (out_of_scope_refusal кейсы).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.generation.llm_client import get_llm_client


@dataclass
class EvalResult:
    question: str
    category: str
    retrieval_hit: bool  # нашёлся ли ожидаемый источник в топ-K
    answer_relevance_score: float  # 0-10, от LLM-judge
    groundedness_score: float  # 0-10, от LLM-judge
    refusal_correct: bool | None  # только для category=out_of_scope_refusal
    generated_answer: str
    latency_ms: float


def check_retrieval_hit(retrieved_pages: list[int], expected_page: int | None) -> bool:
    """
    Для out_of_scope вопросов (expected_page=None) считаем retrieval_hit
    неприменимым — возвращаем True, чтобы не портить агрегированную метрику
    вопросами, где источника в принципе быть не должно.
    """
    if expected_page is None:
        return True
    return expected_page in retrieved_pages


def score_answer_relevance(question: str, answer: str) -> float:
    """
    LLM-as-judge: оценивает отвечает ли answer на question,
    независимо от того, правильный ли это ответ по фактам.
    Пример: если ответ "в документе нет информации" — это релевантный
    ответ на вопрос (модель поняла вопрос и корректно среагировала).
    """
    client = get_llm_client()
    result = client.generate(
        system_prompt=(
            "Ты — судья, который оценивает релевантность ответа вопросу. "
            "Оцени по шкале 0-10, насколько ответ релевантен вопросу "
            "(отвечает ли он вообще на то, что спросили, даже если это "
            "корректный отказ). Ответь ТОЛЬКО числом, без пояснений."
        ),
        user_message=f"Вопрос: {question}\n\nОтвет: {answer}",
        temperature=0.0,
    )
    return _parse_score(result.text)


def score_groundedness(context_chunks: list[str], answer: str) -> float:
    """
    LLM-as-judge: проверяет, что каждое утверждение в answer подкреплено
    контекстом. Это ключевая защита от галлюцинаций — низкий скор значит
    модель добавила факты, которых не было в документе.
    """
    client = get_llm_client()
    context_text = "\n---\n".join(context_chunks)

    result = client.generate(
        system_prompt=(
            "Ты — судья, который проверяет обоснованность ответа. "
            "Оцени по шкале 0-10, насколько ВСЕ утверждения в ответе "
            "подкреплены предоставленным контекстом (10 = полностью "
            "обоснован, 0 = полностью выдуман / не связан с контекстом). "
            "Ответ вида 'в документе нет информации' при отсутствии "
            "релевантного контекста — это 10 (полностью обоснованный отказ). "
            "Ответь ТОЛЬКО числом, без пояснений."
        ),
        user_message=f"Контекст:\n{context_text}\n\nОтвет модели: {answer}",
        temperature=0.0,
    )
    return _parse_score(result.text)


def _parse_score(text: str) -> float:
    """Парсит числовой скор из ответа судьи, с фолбэком при ошибке парсинга."""
    cleaned = text.strip().split()[0] if text.strip() else "0"
    try:
        score = float(cleaned)
        return max(0.0, min(10.0, score))
    except ValueError:
        return 0.0


def aggregate_results(results: list[EvalResult]) -> dict:
    """Сводная статистика по всему eval run — то, что смотришь после каждого прогона."""
    if not results:
        return {}

    n = len(results)
    retrieval_hit_rate = sum(r.retrieval_hit for r in results) / n
    avg_relevance = sum(r.answer_relevance_score for r in results) / n
    avg_groundedness = sum(r.groundedness_score for r in results) / n
    avg_latency = sum(r.latency_ms for r in results) / n

    refusal_cases = [r for r in results if r.refusal_correct is not None]
    refusal_accuracy = (
        sum(r.refusal_correct for r in refusal_cases) / len(refusal_cases)
        if refusal_cases
        else None
    )

    return {
        "num_cases": n,
        "retrieval_hit_rate": round(retrieval_hit_rate, 3),
        "avg_answer_relevance": round(avg_relevance, 2),
        "avg_groundedness": round(avg_groundedness, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "refusal_accuracy": round(refusal_accuracy, 3) if refusal_accuracy is not None else None,
        "cases_below_groundedness_threshold": [
            r.question for r in results if r.groundedness_score < 7.0
        ],
    }
