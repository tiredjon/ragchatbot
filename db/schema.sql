-- Схема для pgvector в Supabase.
-- Выполнить один раз в SQL Editor твоего Supabase проекта.

-- Расширение pgvector (в Supabase обычно уже доступно, но на всякий случай)
create extension if not exists vector;

create table if not exists document_chunks (
    id bigint generated always as identity primary key,
    document_id uuid not null,
    chunk_index int not null,
    source_filename text not null,
    page_number int not null,
    text text not null,
    embedding vector(768),  -- text-embedding-004 у Gemini выдаёт 768-мерный вектор
    approx_tokens int,
    created_at timestamptz default now()
);

-- Индекс для быстрого поиска по document_id (фильтрация перед векторным поиском)
create index if not exists idx_document_chunks_document_id
    on document_chunks (document_id);

-- HNSW индекс для быстрого приближённого поиска по косинусному расстоянию.
-- Без этого индекса на больших объёмах данных запросы будут делать
-- полный перебор (exact search), что медленно при росте базы.
create index if not exists idx_document_chunks_embedding
    on document_chunks using hnsw (embedding vector_cosine_ops);

-- Опционально: таблица для метаданных загруженных документов
-- (имя файла, дата загрузки, кто загрузил — если добавишь auth)
create table if not exists documents (
    id uuid primary key,
    filename text not null,
    num_chunks int not null,
    uploaded_at timestamptz default now()
);
