"""
Раннер evals. Запуск: python -m evals.run_evals

Что делает:
1. Читает evals/dataset.jsonl (список тестовых вопросов)
2. Для каждого — вызывает retrieval + generation pipeline напрямую
   (не через HTTP, а вызывая функции — быстрее и не нужен запущенный сервер)
3. Считает метрики через evals/metrics.py
4. Печатает сводку + сохраняет детальный отчёт в evals/results/

ВАЖНО: перед запуском нужно:
1. Загрузить тестовый документ через /upload (или ingestion pipeline напрямую)
2. Проставить реальный document_id в dataset.jsonl вместо REPLACE_WITH_REAL_DOCUMENT_ID
3. Вручную определить expected_source_page для каждого вопроса, посмотрев
   в реальный документ — это ручная работа, но именно она делает eval
   осмысленным, а не постановочным
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.generation.prompt_templates import build_system_prompt, build_user_message
from app.generation.llm_client import get_llm_client
from app.observability.tracer import QueryTrace
from app.retrieval.query_processor import process_query
from app.retrieval.retriever import retrieve
from evals.metrics import (
    EvalResult,
    aggregate_results,
    check_retrieval_hit,
    score_answer_relevance,
    score_groundedness,
)

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset() -> list[dict]:
    cases = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_single_case(case: dict) -> EvalResult:
    document_id = case["document_id"]
    question = case["question"]

    if document_id == "REPLACE_WITH_REAL_DOCUMENT_ID":
        raise ValueError(
            "dataset.jsonl всё ещё содержит плейсхолдер document_id. "
            "Загрузи тестовый документ и подставь реальный ID."
        )

    trace = QueryTrace(query_raw=question)
    processed = process_query(question)
    trace.query_language_detected = processed.detected_language

    top_chunks = retrieve(query=processed.normalized, document_id=document_id, trace=trace)
    retrieved_pages = [c.page_number for c in top_chunks]

    client = get_llm_client()
    system_prompt = build_system_prompt(processed.detected_language)
    user_message = build_user_message(processed.normalized, top_chunks)

    trace.start_stage()
    generation_result = client.generate(system_prompt=system_prompt, user_message=user_message)
    trace.generation_latency_ms = trace.elapsed_ms()

    answer = generation_result.text

    retrieval_hit = check_retrieval_hit(retrieved_pages, case.get("expected_source_page"))
    relevance = score_answer_relevance(question, answer)
    groundedness = score_groundedness([c.text for c in top_chunks], answer)

    refusal_correct = None
    if case.get("category") == "out_of_scope_refusal":
        expected_markers = case.get("expected_answer_contains", [])
        refusal_correct = any(marker.lower() in answer.lower() for marker in expected_markers)

    return EvalResult(
        question=question,
        category=case.get("category", "uncategorized"),
        retrieval_hit=retrieval_hit,
        answer_relevance_score=relevance,
        groundedness_score=groundedness,
        refusal_correct=refusal_correct,
        generated_answer=answer,
        latency_ms=trace.total_latency_ms(),
    )


def main() -> None:
    cases = load_dataset()
    print(f"Загружено {len(cases)} тест-кейсов из {DATASET_PATH}")

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['question'][:60]}...")
        try:
            result = run_single_case(case)
            results.append(result)
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            continue

    if not results:
        print("Ни один кейс не выполнился успешно. Проверь document_id в dataset.jsonl.")
        sys.exit(1)

    summary = aggregate_results(results)

    print("\n--- Сводка ---")
    for key, value in summary.items():
        print(f"{key}: {value}")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"eval_run_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "summary": summary,
        "detailed_results": [
            {
                "question": r.question,
                "category": r.category,
                "retrieval_hit": r.retrieval_hit,
                "answer_relevance_score": r.answer_relevance_score,
                "groundedness_score": r.groundedness_score,
                "refusal_correct": r.refusal_correct,
                "generated_answer": r.generated_answer,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nПодробный отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
