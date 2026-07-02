"""
Юнит-тесты для логики, которая не требует API-вызовов или БД
(chunking, language detection). Retrieval/generation с реальными
API-вызовами тестируются через evals/run_evals.py, а не pytest —
это осознанное разделение: юнит-тесты для чистой логики,
evals для end-to-end качества.

Запуск: pytest tests/
"""
from app.ingestion.chunker import _split_into_sentences, chunk_pages
from app.ingestion.loader import RawPage
from app.retrieval.query_processor import process_query


def test_split_into_sentences_basic():
    text = "Это первое предложение. Это второе предложение. А это третье!"
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3


def test_split_into_sentences_handles_empty():
    assert _split_into_sentences("") == []
    assert _split_into_sentences("   ") == []


def test_chunk_pages_preserves_page_number():
    pages = [
        RawPage(text="Первое предложение. " * 50, page_number=1, source_filename="test.pdf"),
        RawPage(text="Второе предложение. " * 50, page_number=2, source_filename="test.pdf"),
    ]
    chunks = chunk_pages(pages)

    assert len(chunks) > 0
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_chunk_pages_empty_input():
    assert chunk_pages([]) == []


def test_process_query_detects_russian():
    result = process_query("Какой размер комиссии?")
    assert result.detected_language == "ru"


def test_process_query_detects_uzbek():
    result = process_query("Bu hujjatda qanday ma'lumot bor?")
    assert result.detected_language == "uz"


def test_process_query_unknown_for_ambiguous_latin():
    result = process_query("xyz abc")
    assert result.detected_language == "unknown"
