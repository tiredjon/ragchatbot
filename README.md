# Multilingual RAG Agent (UZ/RU)

RAG-агент без фронтенда, так как цель этого проекта была натренировать руку для разработки RAG-модели. Он отвечает на вопросы по загруженным документам (PDF/DOCX/TXT)
на узбекском и русском языках. Построен с акцентом на production reliability:
evals, observability, source attribution — а не только "работающий happy path".

## Архитектура

```
Upload -> Loader -> Chunker -> Embedder (Gemini) -> pgvector (Supabase)
Query  -> Language detection -> Vector search -> LLM Rerank -> Generation -> Answer + Sources
```

Каждый запрос трейсится (`app/observability/tracer.py`) — латентность
каждого этапа, какие чанки нашлись, с каким скором, сколько токенов
потрачено. Это то, что позволяет дебажить "почему агент ответил плохо"
постфактум, а не гадать.

## Почему так, а не иначе

- **Rerank поверх vector search** — чистый cosine similarity находит
  "похожее по смыслу", но не всегда самое релевантное. LLM-rerank —
  второй, более точный проход по top-K кандидатам.
- **Explicit language detection** — мультиязычные боты часто отвечают
  не на том языке, на котором спросили. Детектим язык явно и передаём
  в system prompt.
- **Groundedness eval через LLM-as-judge** — главная защита от
  галлюцинаций в RAG не "промпт лучше написать", а измеримая метрика,
  которую можно отслеживать между версиями промпта.
- **Structured JSON logging** — каждый лог пригоден для парсинга,
  не текстовый print().

## Известные ограничения

- Rerank через LLM-scoring, а не dedicated cross-encoder модель
  (например `bge-reranker`) — проще в сетапе, но медленнее и дороже
  по токенам. В реальном проде с большим трафиком — заменить.
- Chunking по предложениям через regex, а не полноценный sentence
  tokenizer (spacy/nltk с UZ/RU моделью) — для большинства документов
  достаточно, но на текстах с нестандартной пунктуацией может рвать
  предложения не в тех местах.
- DOCX парсинг группирует параграфы эвристикой (15 параграфов = "страница"),
  так как DOCX не хранит реальную пагинацию — в отличие от PDF.
- Rate limits бесплатного тира Gemini API могут быть узким местом
  при батч-обработке больших документов или прогоне evals — при
  необходимости добавить exponential backoff в `llm_client.py`.

## Setup

```bash
# 1. Клонировать зависимости
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Настроить .env
cp .env.example .env
# Заполнить GEMINI_API_KEY (https://aistudio.google.com/apikey — бесплатно)
# Заполнить SUPABASE_URL, SUPABASE_SERVICE_KEY, DATABASE_URL

# 3. Создать таблицы в Supabase
# Открыть SQL Editor в Supabase dashboard, выполнить db/schema.sql

# 4. Запустить сервер
uvicorn app.main:app --reload --port 8000

# 5. Открыть Swagger UI
# http://localhost:8000/docs
```

## Использование

1. `POST /upload` — загрузить документ (multipart/form-data, поле `file`)
   → получить `document_id`
2. `POST /query` — задать вопрос: `{"document_id": "...", "question": "..."}`
   → получить ответ с указанием источника (файл + страница)

## Запуск evals

```bash
# 1. Загрузить тестовый документ через /upload, скопировать document_id
# 2. Подставить document_id в evals/dataset.jsonl
# 3. Определить expected_source_page для каждого вопроса вручную
#    (посмотреть в документ, на какой странице реальный ответ)
python -m evals.run_evals
```

Метрики, которые считаются:
- **Retrieval Hit Rate** — нашёлся ли ожидаемый источник в топ-K
- **Answer Relevance** — отвечает ли ответ на заданный вопрос (LLM-judge)
- **Groundedness** — опирается ли ответ только на контекст, без галлюцинаций (LLM-judge)
- **Refusal Accuracy** — корректно ли агент отказывается отвечать на
  вопросы не по документу

## Тесты

```bash
pytest tests/
```

Юнит-тесты покрывают чистую логику (chunking, language detection) без
API-вызовов. Retrieval/generation качество проверяется через evals, не pytest —
осознанное разделение "правильность кода" vs "качество ответов".

## Стек

FastAPI · Gemini API (генерация + эмбеддинги) · Supabase / pgvector · psycopg3
