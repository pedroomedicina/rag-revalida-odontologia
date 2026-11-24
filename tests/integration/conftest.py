"""Shared fixtures for tests that need a real Postgres (RAG_TEST_DSN)."""

from __future__ import annotations

import os
import uuid

import pytest

from rag.text_search import ensure_text_search_config

psycopg = pytest.importorskip("psycopg")


@pytest.fixture(scope="module")
def conn():
    """Autocommit connection whose search_path is a throwaway schema, dropped afterwards."""
    dsn = os.environ.get("RAG_TEST_DSN")
    if not dsn:
        pytest.skip("RAG_TEST_DSN not set")
    try:
        connection = psycopg.connect(dsn, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"cannot connect to RAG_TEST_DSN: {exc}")
    schema = f"rag_test_{uuid.uuid4().hex[:8]}"
    with connection.cursor() as cur:
        ensure_text_search_config(cur)  # database-level, idempotent
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}, public")
    yield connection
    with connection.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    connection.close()


@pytest.fixture(scope="module")
def require_pgvector(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        if cur.fetchone() is None:
            pytest.skip("pgvector not installed on this Postgres")
