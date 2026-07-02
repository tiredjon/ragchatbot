"""
Точка входа приложения.

Запуск локально: uvicorn app.main:app --reload --port 8000
После запуска: http://localhost:8000/docs — автогенерируемая Swagger UI,
где можно потыкать эндпоинты руками без Postman/curl.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Multilingual RAG Agent",
    description="RAG-агент с поддержкой UZ/RU, загрузкой документов и observability.",
    version="0.1.0",
)

# CORS открыт для разработки. В проде — сузить allow_origins до конкретных доменов.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
