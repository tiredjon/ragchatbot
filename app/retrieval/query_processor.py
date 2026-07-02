"""
Предобработка запроса пользователя перед retrieval.

Зачем детектить язык явно, а не просто кидать всё в один эмбеддинг:
1. Для observability — важно логировать на каком языке спросили,
   иначе непонятно почему в проде вдруг просела точность на части трафика
2. Для генерации — system prompt инструктирует модель отвечать
   на том же языке, на котором спросили (частая проблема мультиязычных
   ботов — ответ на русском на узбекский вопрос)

Детекция намеренно простая (эвристика по алфавиту и стоп-словам),
а не через тяжёлую ML-библиотеку — для UZ (латиница) vs RU (кириллица)
этого более чем достаточно и не тянет лишнюю зависимость.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# частые узбекские служебные слова/окончания на латинице,
# которых не бывает в русском/английском тексте
_UZBEK_MARKERS = {
    "va", "bilan", "uchun", "qanday", "nima", "qachon", "qayerda",
    "bo'lsa", "kerak", "haqida", "yoki", "lekin", "ham",
}

_CYRILLIC_PATTERN = re.compile(r"[а-яА-ЯёЁ]")


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    detected_language: str  # "ru" | "uz" | "unknown"


def process_query(raw_query: str) -> ProcessedQuery:
    normalized = raw_query.strip()
    language = _detect_language(normalized)

    return ProcessedQuery(
        original=raw_query,
        normalized=normalized,
        detected_language=language,
    )


def _detect_language(text: str) -> str:
    if _CYRILLIC_PATTERN.search(text):
        return "ru"

    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    if words & _UZBEK_MARKERS:
        return "uz"

    # Латиница без узбекских маркеров — неоднозначно (может быть английский
    # или узбекский без характерных слов в коротком запросе).
    # Возвращаем "unknown", downstream код должен на это реагировать
    # (например, спросить пользователя или дать ответ на обоих).
    if re.search(r"[a-zA-Z]", text):
        return "unknown"

    return "unknown"
