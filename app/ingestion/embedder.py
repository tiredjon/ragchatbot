"""
Эмбеддинг чанков и запись в pgvector (Supabase).

Схема таблицы — см. db/schema.sql. Коротко: каждая строка это один чанк
с его вектором, текстом и метаданными источника (файл, страница, документ).

document_id группирует чанки одной загрузки — это нужно, чтобы можно было
удалить/переиндексировать один документ, не трогая остальные в базе.
"""
from __future__ import annotations

import uuid

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.generation.llm_client import get_llm_client
from app.ingestion.chunker import Chunk
from app.observability.logger import get_logger

logger = get_logger(__name__)


def embed_and_store_chunks(chunks: list[Chunk], document_id: str | None = None) -> str:
    """
    Эмбеддит список чанков и сохраняет их в pgvector.
    Возвращает document_id (генерируется, если не передан) —
    он понадобится клиенту, чтобы потом фильтровать поиск по этому документу.
    """
    if document_id is None:
        document_id = str(uuid.uuid4())

    settings = get_settings()
    client = get_llm_client()

    logger.info(
        "embedding_started",
        extra={"document_id": document_id, "num_chunks": len(chunks)},
    )

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                embedding_result = client.embed_text(chunk.text, task_type="retrieval_document")

                cur.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, chunk_index, source_filename, page_number,
                         text, embedding, approx_tokens)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk.chunk_index,
                        chunk.source_filename,
                        chunk.page_number,
                        chunk.text,
                        embedding_result.vector,
                        chunk.approx_tokens,
                    ),
                )
        conn.commit()

    logger.info(
        "embedding_completed",
        extra={"document_id": document_id, "num_chunks": len(chunks)},
    )

    return document_id


def delete_document(document_id: str) -> int:
    """Удаляет все чанки документа. Возвращает число удалённых строк."""
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )
            deleted = cur.rowcount
        conn.commit()
    return deleted
