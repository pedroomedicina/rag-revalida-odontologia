"""Pure retrieval helpers (no database, no models)."""

from __future__ import annotations

import pytest

from rag.retrieval_smoke_test import parse_cli as parse_smoke_cli
from rag.retrieve_hybrid import (
    apply_quality_prior,
    build_generator_payload,
    format_citation,
    hybrid_fuse,
    normalize_query,
    validate_table_name,
)


def hit(chunk_id: str, **extra) -> dict[str, object]:
    return {"chunk_id": chunk_id, "doc_id": "D", "page_start": 1, "page_end": 1, **extra}


def test_normalize_query():
    assert normalize_query("  ¿Qué   es  el EDTA? ¡Ya! ") == "qué es el edta? ya!"


def test_hybrid_fuse_rewards_agreement():
    merged = hybrid_fuse(
        vector_hits=[hit("a"), hit("b")],
        keyword_hits=[hit("b"), hit("c")],
        vector_weight=0.55,
        keyword_weight=0.45,
        rrf_k=60,
    )
    assert [m["chunk_id"] for m in merged] == ["b", "a", "c"]
    b = merged[0]
    assert b["vector_rank"] == 2 and b["keyword_rank"] == 1
    assert b["hybrid_score"] == pytest.approx(0.55 / 62 + 0.45 / 61)


def test_hybrid_fuse_handles_empty_keyword_branch():
    merged = hybrid_fuse([hit("a")], [], 0.55, 0.45, 60)
    assert merged[0]["hybrid_score"] == pytest.approx(0.55 / 61)


@pytest.mark.parametrize(
    ("quality", "prior"),
    [("good", 0.05), ("usable_with_caution", -0.02), ("poor", -0.10), ("", 0.0)],
)
def test_quality_prior(quality, prior):
    assert apply_quality_prior({"ocr_quality": quality}) == prior


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [(3, 3, "[D p.3]"), (3, 5, "[D pp.3-5]"), (0, 0, "[D]")],
)
def test_format_citation(start, end, expected):
    assert format_citation({"doc_id": "D", "page_start": start, "page_end": end}) == expected


def test_generator_payload_lists_citations():
    payload = build_generator_payload("¿Qué?", [hit("a", title="Guía", text="Texto.")])
    assert "#### Chunk 1 [D p.1]" in payload
    assert "Responde solo con base en la evidencia" in payload


def test_table_name_rejects_injection():
    with pytest.raises(ValueError):
        validate_table_name("rag_chunks; DROP TABLE x")


def test_smoke_test_forwards_unknown_flags():
    args = parse_smoke_cli(["--min-results", "2", "--dsn", "postgresql://x", "--table", "t"])
    assert args.min_results == 2
    assert args.extra_args == ["--dsn", "postgresql://x", "--table", "t"]


def test_smoke_test_still_accepts_double_dash_form():
    args = parse_smoke_cli(["--", "--dsn", "postgresql://x"])
    assert args.extra_args == ["--dsn", "postgresql://x"]
