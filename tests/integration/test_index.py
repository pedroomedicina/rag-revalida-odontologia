"""Index replacement semantics (needs Postgres with pgvector)."""

from __future__ import annotations

import pytest

from rag.build_retrieval_index import ChunkRecord, ensure_schema, replace_chunks

pytestmark = pytest.mark.integration

TABLE = "rag_chunks_replace"


def record(doc_id: str, n: int) -> ChunkRecord:
    text = f"Texto del fragmento {n} del documento {doc_id}."
    return ChunkRecord(
        chunk_id=f"{doc_id}-ch-{n:04d}",
        doc_id=doc_id,
        title="Guía",
        filename=f"{doc_id}.pdf",
        page_start=n,
        page_end=n,
        section_title="",
        content_type="conceptual",
        ocr_quality="good",
        extraction_method="pypdf",
        token_estimate=len(text.split()),
        text=text,
    )


def vectors(count: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in range(count)]


def rows(conn) -> dict[str, list[str]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT doc_id, chunk_id FROM {TABLE} ORDER BY chunk_id")
        out: dict[str, list[str]] = {}
        for doc_id, chunk_id in cur.fetchall():
            out.setdefault(doc_id, []).append(chunk_id)
        return out


@pytest.fixture
def table(conn, require_pgvector):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
        ensure_schema(cur, TABLE, embedding_dim=3)
        initial = [record("A", n) for n in (1, 2, 3)] + [record("B", n) for n in (1, 2)]
        replace_chunks(cur, TABLE, initial, vectors(len(initial)))
    return TABLE


def test_rechunked_document_loses_old_chunk_ids(conn, table):
    with conn.cursor() as cur:
        counts = replace_chunks(cur, table, [record("A", 1)], vectors(1))
    assert rows(conn) == {"A": ["A-ch-0001"], "B": ["B-ch-0001", "B-ch-0002"]}
    assert counts == {"documents": 1, "deleted_rows": 3, "pruned_rows": 0, "inserted_rows": 1}


def test_documents_not_in_the_run_are_kept_without_prune(conn, table):
    with conn.cursor() as cur:
        replace_chunks(cur, table, [record("C", 1)], vectors(1))
    assert set(rows(conn)) == {"A", "B", "C"}


def test_prune_removes_documents_missing_from_the_run(conn, table):
    with conn.cursor() as cur:
        counts = replace_chunks(cur, table, [record("A", 1)], vectors(1), prune=True)
    assert rows(conn) == {"A": ["A-ch-0001"]}
    assert counts["pruned_rows"] == 2


def test_replacement_is_atomic_on_failure(conn, table):
    bad = [record("A", 1)]
    with pytest.raises(ValueError):  # zip(strict=True): vectors don't match records
        with conn.transaction():
            with conn.cursor() as cur:
                replace_chunks(cur, table, bad, vectors(2))
    assert rows(conn)["A"] == ["A-ch-0001", "A-ch-0002", "A-ch-0003"]
