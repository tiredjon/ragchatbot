"""
Загрузка документов: PDF / DOCX / TXT -> сырой текст с метаданными.

Важно: мы сохраняем номер страницы/раздела в метаданных каждого куска текста —
это нужно для observability (чтобы в логах было видно ИЗ КАКОЙ страницы
пришёл ответ) и для evals (проверка, что retrieval нашёл правильный источник).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pypdf
from docx import Document as DocxDocument


@dataclass
class RawPage:
    """Один логический блок документа (страница PDF или раздел docx)."""
    text: str
    page_number: int
    source_filename: str
    metadata: dict = field(default_factory=dict)


class UnsupportedFileTypeError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


def load_document(file_bytes: bytes, filename: str) -> list[RawPage]:
    """
    Главная точка входа. Определяет тип файла по расширению и
    диспатчит на соответствующий парсер.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        pages = _load_pdf(file_bytes, filename)
    elif suffix == ".docx":
        pages = _load_docx(file_bytes, filename)
    elif suffix == ".txt":
        pages = _load_txt(file_bytes, filename)
    else:
        raise UnsupportedFileTypeError(
            f"Формат '{suffix}' не поддерживается. Используй PDF, DOCX или TXT."
        )

    # фильтруем пустые/мусорные страницы (сканы без OCR текста и т.п.)
    pages = [p for p in pages if p.text.strip()]

    if not pages:
        raise EmptyDocumentError(
            "Не удалось извлечь текст из документа. "
            "Возможно это скан без текстового слоя (нужен OCR)."
        )

    return pages


def _load_pdf(file_bytes: bytes, filename: str) -> list[RawPage]:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            RawPage(
                text=text,
                page_number=i,
                source_filename=filename,
                metadata={"total_pages": len(reader.pages)},
            )
        )
    return pages


def _load_docx(file_bytes: bytes, filename: str) -> list[RawPage]:
    """
    DOCX не имеет "страниц" в файловой структуре (пагинация — это рендеринг),
    поэтому группируем по секциям через разрывы страниц/крупные заголовки.
    Для простоты: каждые N параграфов = один логический "блок".
    """
    doc = DocxDocument(io.BytesIO(file_bytes))
    pages = []
    current_block: list[str] = []
    block_number = 1
    paragraphs_per_block = 15  # эвристика; можно тюнить

    for para in doc.paragraphs:
        if para.text.strip():
            current_block.append(para.text)

        if len(current_block) >= paragraphs_per_block:
            pages.append(
                RawPage(
                    text="\n".join(current_block),
                    page_number=block_number,
                    source_filename=filename,
                )
            )
            current_block = []
            block_number += 1

    if current_block:
        pages.append(
            RawPage(
                text="\n".join(current_block),
                page_number=block_number,
                source_filename=filename,
            )
        )

    return pages


def _load_txt(file_bytes: bytes, filename: str) -> list[RawPage]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [RawPage(text=text, page_number=1, source_filename=filename)]
