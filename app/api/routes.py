"""
API роуты. Три эндпоинта:
- POST /upload  — загрузка документа, ingestion pipeline
- POST /query   — вопрос по загруженному документу
- GET  /health  — healthcheck

FastAPI-специфика для тех, кто первый раз видит фреймворк:
- @router.post(...) — декоратор регистрирует функцию как обработчик роута
- Параметры функции с типами (напр. file: UploadFile) FastAPI парсит
  автоматически из запроса — не нужно вручную разбирать request body
- response_model — Pydantic модель, которая описывает форму ответа
  (и заодно валидирует его перед отправкой)
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.generation.llm_client import get_llm_client
from app.generation.prompt_templates import build_system_prompt, build_user_message
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_and_store_chunks
from app.ingestion.loader import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)
from app.observability.logger import get_logger
from app.observability.tracer import QueryTrace
from app.retrieval.query_processor import process_query
from app.retrieval.retriever import retrieve

logger = get_logger(__name__)
router = APIRouter()


# --- Response/Request models ---

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int


class QueryRequest(BaseModel):
    document_id: str
    question: str
    mode: str = "rag"  # "rag" (строгий, только факты) | "chat" (свободное обсуждение)


class SourceCitation(BaseModel):
    filename: str
    page: int
    relevance_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    detected_language: str
    trace_id: str
    latency_ms: float


# --- Routes ---

@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой ({size_mb:.1f}MB). Лимит: {settings.max_upload_mb}MB.",
        )

    try:
        pages = load_document(file_bytes, file.filename)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmptyDocumentError as e:
        raise HTTPException(status_code=422, detail=str(e))

    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="После обработки документа не осталось текста для индексации.",
        )

    document_id = embed_and_store_chunks(chunks)

    logger.info(
        "document_uploaded",
        extra={
            "document_id": document_id,
            "uploaded_filename": file.filename,
            "num_chunks": len(chunks),
        },
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        num_chunks=len(chunks),
    )


@router.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest) -> QueryResponse:
    trace = QueryTrace(query_raw=request.question)

    try:
        processed = process_query(request.question)
        trace.query_language_detected = processed.detected_language

        top_chunks = retrieve(
            query=processed.normalized,
            document_id=request.document_id,
            trace=trace,
        )

        if not top_chunks:
            trace.error = "no_chunks_found"
            trace.emit()
            raise HTTPException(
                status_code=404,
                detail="Документ не найден или не содержит проиндексированных данных.",
            )

        client = get_llm_client()
        system_prompt = build_system_prompt(processed.detected_language, mode=request.mode)
        user_message = build_user_message(processed.normalized, top_chunks)

        trace.start_stage()
        result = client.generate(system_prompt=system_prompt, user_message=user_message)
        trace.generation_latency_ms = trace.elapsed_ms()
        trace.generation_input_tokens = result.input_tokens
        trace.generation_output_tokens = result.output_tokens
        trace.final_answer = result.text

        trace.emit()

        return QueryResponse(
            answer=result.text,
            sources=[
                SourceCitation(
                    filename=c.source_filename,
                    page=c.page_number,
                    relevance_score=c.rerank_score,
                )
                for c in top_chunks
            ],
            detected_language=processed.detected_language,
            trace_id=trace.trace_id,
            latency_ms=trace.total_latency_ms(),
        )

    except HTTPException:
        raise
    except Exception as e:
        trace.error = str(e)
        trace.emit()
        logger.exception("query_failed", extra={"trace_id": trace.trace_id})
        raise HTTPException(status_code=500, detail="Внутренняя ошибка при обработке запроса.")
