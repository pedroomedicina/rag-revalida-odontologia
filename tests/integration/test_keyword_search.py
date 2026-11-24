"""Keyword search against a real Postgres.

Run with RAG_TEST_DSN set, e.g.:
    RAG_TEST_DSN="postgresql:///rag_revalida_test?host=/tmp" uv run pytest -m integration

Keyword-only tests need just Postgres (with the unaccent contrib extension). Tests that
build the full schema also need pgvector; they skip when it isn't installed. CI runs all
of them against a pgvector service container.
"""

from __future__ import annotations

import pytest

from rag.build_retrieval_index import ChunkRecord, ensure_schema, upsert_batch
from rag.retrieve_hybrid import fetch_keyword_hits, fetch_vector_hits, normalize_query
from rag.text_search import (
    SEARCH_TSV_EXPRESSION,
    TS_CONFIG,
    ensure_search_column,
    search_tsv_expression_of,
)

pytestmark = pytest.mark.integration

CHUNKS = {
    "calcio": (
        "MEDICACIÓN INTRACANAL",
        "El hidróxido de calcio se indica como medicación intracanal entre sesiones cuando "
        "existe necrosis pulpar con lesión periapical.",
    ),
    "irrigacion": (
        "IRRIGACIÓN",
        "El hipoclorito de sodio es el irrigante principal durante la preparación "
        "químico-mecánica. Disuelve tejido orgánico.",
    ),
    "obturacion": (
        "OBTURACIÓN",
        "La obturación busca un sellado tridimensional que impida la recontaminación.",
    ),
}


def create_keyword_table(cur, table: str, tsv_expression: str) -> None:
    cur.execute(
        f"""
        CREATE TABLE {table} (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, title TEXT NOT NULL,
            filename TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
            section_title TEXT NOT NULL, content_type TEXT NOT NULL, ocr_quality TEXT NOT NULL,
            text TEXT NOT NULL,
            search_tsv tsvector GENERATED ALWAYS AS ({tsv_expression}) STORED
        )
        """
    )
    for chunk_id, (section, text) in CHUNKS.items():
        cur.execute(
            f"INSERT INTO {table} VALUES (%s, 'D', 'Guía', 'guia.pdf', 1, 1, %s, "
            "'conceptual', 'good', %s)",
            (chunk_id, section, text),
        )


@pytest.fixture(scope="module")
def table(conn):
    with conn.cursor() as cur:
        create_keyword_table(cur, "kw_chunks", SEARCH_TSV_EXPRESSION)
    return "kw_chunks"


def search(conn, table: str, question: str, limit: int = 10) -> list[str]:
    with conn.cursor() as cur:
        hits = fetch_keyword_hits(cur, table, normalize_query(question), limit)
    return [hit["chunk_id"] for hit in hits]


def test_natural_question_matches_with_words_missing_from_chunk(conn, table):
    # "cuándo" and "endodoncia" are not in the chunk; plainto_tsquery returned nothing.
    question = "¿Cuándo se indica el hidróxido de calcio en endodoncia?"
    assert search(conn, table, question)[0] == "calcio"


def test_question_without_accents_matches_accented_text(conn, table):
    assert search(conn, table, "medicacion intracanal") == ["calcio"]
    assert search(conn, table, "hidroxido de calcio") == ["calcio"]


def test_accented_question_matches(conn, table):
    assert search(conn, table, "irrigación con hipoclorito") == ["irrigacion"]


def test_chunk_matching_more_words_ranks_first(conn, table):
    hits = search(conn, table, "hipoclorito irrigante sellado")
    assert hits[:2] == ["irrigacion", "obturacion"]


def test_stopword_only_question_returns_nothing(conn, table):
    assert search(conn, table, "¿qué es el?") == []


def test_section_title_is_searchable(conn, table):
    assert search(conn, table, "obturacion") == ["obturacion"]


def test_old_spanish_column_is_migrated(conn):
    old_expression = SEARCH_TSV_EXPRESSION.replace(f"'{TS_CONFIG}'", "'spanish'")
    with conn.cursor() as cur:
        create_keyword_table(cur, "kw_legacy", old_expression)
        assert TS_CONFIG not in search_tsv_expression_of(cur, "kw_legacy")
        assert ensure_search_column(cur, "kw_legacy") is True
        assert TS_CONFIG in search_tsv_expression_of(cur, "kw_legacy")
        assert ensure_search_column(cur, "kw_legacy") is False  # idempotent
    assert search(conn, "kw_legacy", "medicacion intracanal") == ["calcio"]


@pytest.fixture(scope="module")
def full_table(conn, require_pgvector):
    with conn.cursor() as cur:
        ensure_schema(cur, "rag_chunks_full", embedding_dim=3)
        records, vectors = [], []
        for index, (chunk_id, (section, text)) in enumerate(CHUNKS.items()):
            records.append(
                ChunkRecord(
                    chunk_id,
                    "D",
                    "Guía",
                    "guia.pdf",
                    1,
                    1,
                    section,
                    "conceptual",
                    "good",
                    "pypdf",
                    len(text.split()),
                    text,
                )
            )
            vectors.append([1.0 if i == index else 0.0 for i in range(3)])
        upsert_batch(cur, "rag_chunks_full", records, vectors)
    return "rag_chunks_full"


def test_full_schema_keyword_and_vector_search(conn, full_table):
    assert search(conn, full_table, "medicacion intracanal") == ["calcio"]
    with conn.cursor() as cur:
        hits = fetch_vector_hits(cur, full_table, "[0,1,0]", 1)
    assert hits[0]["chunk_id"] == "irrigacion"


def test_ensure_schema_is_idempotent(conn, full_table):
    with conn.cursor() as cur:
        ensure_schema(cur, full_table, embedding_dim=3)
        assert TS_CONFIG in search_tsv_expression_of(cur, full_table)
