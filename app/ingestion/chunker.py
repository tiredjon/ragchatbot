"""
Разбиение текста на чанки для эмбеддинга.

Почему не просто "резать каждые 500 символов":
- Фиксированная резка по символам рвёт предложения посередине,
  что портит качество эмбеддинга (модель получает обрывок мысли).
- Мы режем по абзацам/предложениям и группируем их в чанки нужного
  размера, с overlap между соседними чанками — это стандартная практика,
  которая уменьшает потерю контекста на границах чанков.

Токены считаем приблизительно (по словам * 1.3 для UZ/RU текста),
без подключения полноценного токенайзера — этого достаточно для чанкинга.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_settings
from app.ingestion.loader import RawPage


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_filename: str
    page_number: int
    approx_tokens: int


def _approx_token_count(text: str) -> int:
    """Грубая оценка числа токенов. Для UZ/RU 1 слово ~= 1.3 токена."""
    word_count = len(text.split())
    return int(word_count * 1.3)


def _split_into_sentences(text: str) -> list[str]:
    """
    Разбивка по предложениям. Учитываем что в UZ/RU тексте точки
    также используются в сокращениях (см., т.д.) — упрощённый regex,
    для продакшена стоит заменить на nltk/spacy с UZ/RU моделью.
    """
    # разбиваем по .!? за которыми следует пробел+заглавная буква или конец строки
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁЎҚҒҲ])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(pages: list[RawPage]) -> list[Chunk]:
    """
    Главная точка входа. Проходит по всем страницам документа,
    режет на чанки с overlap, сохраняя привязку к исходной странице.
    """
    settings = get_settings()
    target_tokens = settings.chunk_size_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    chunks: list[Chunk] = []
    chunk_index = 0

    for page in pages:
        sentences = _split_into_sentences(page.text)
        if not sentences:
            continue

        current_sentences: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = _approx_token_count(sentence)

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

            is_last_sentence = i == len(sentences) - 1
            if current_tokens >= target_tokens or is_last_sentence:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        source_filename=page.source_filename,
                        page_number=page.page_number,
                        approx_tokens=current_tokens,
                    )
                )
                chunk_index += 1

                # overlap: оставляем последние N токенов предложений
                # как начало следующего чанка, чтобы не терять контекст на границе
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    overlap_count += _approx_token_count(s)
                    overlap_sentences.insert(0, s)
                    if overlap_count >= overlap_tokens:
                        break

                current_sentences = overlap_sentences.copy()
                current_tokens = overlap_count

            i += 1

    return chunks
