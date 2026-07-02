"""
Структурированное логирование в JSON.

Почему не просто print() или стандартный logging с текстовым форматом:
- JSON-логи легко парсить/фильтровать (grep по полю, а не по тексту)
- В проде их можно скормить в любую систему (Datadog, CloudWatch, etc)
- Каждый запрос к RAG-агенту должен оставлять след: что искали,
  что нашли, что ответили, сколько это заняло — без этого невозможно
  дебажить "почему агент ответил плохо" постфактум
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.config import get_settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # добавляем кастомные поля из extra={...}
        standard_keys = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in standard_keys and key != "message":
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        settings = get_settings()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False

    return logger
