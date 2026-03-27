#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

from dotenv import load_dotenv

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval (pgvector + Postgres FTS) with reranking."
    )
    parser.add_argument("--query", required=True, help="User question in Spanish")
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres DSN (or set DATABASE_URL)",
    )
    parser.add_argument("--table", default="rag_chunks", help="Chunks table name")
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-base",
        help="SentenceTransformer model used for vector search",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        help="Cross-encoder model for final rerank",
    )
    parser.add_argument("--k-vector", type=int, default=30)
    parser.add_argument("--k-keyword", type=int, default=30)
    parser.add_argument("--k-merge", type=int, default=40)
    parser.add_argument("--k-final", type=int, default=8)
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF dampening constant",
    )
    parser.add_argument("--vector-weight", type=float, default=0.55)
    parser.add_argument("--keyword-weight", type=float, default=0.45)
    parser.add_argument(
        "--prompt-out",
        type=Path,
        default=None,
        help="Optional path to write generator-ready evidence prompt",
    )
    return parser


def validate_table_name(table: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"Invalid table name: {table}")
    return table


def normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def to_vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def fetch_vector_hits(cursor, table: str, query_vector_literal: str, limit: int) -> List[Dict[str, object]]:
    cursor.execute(
        f"""
        SELECT
            chunk_id, doc_id, title, filename, page_start, page_end,
            section_title, content_type, ocr_quality, text,
            (1 - (embedding <=> %s::vector)) AS vector_score
        FROM {table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_vector_literal, query_vector_literal, limit),
    )
    rows = cursor.fetchall()
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def fetch_keyword_hits(cursor, table: str, query: str, limit: int) -> List[Dict[str, object]]:
    cursor.execute(
        f"""
        SELECT
            chunk_id, doc_id, title, filename, page_start, page_end,
            section_title, content_type, ocr_quality, text,
            ts_rank_cd(search_tsv, plainto_tsquery('spanish', %s)) AS keyword_score
        FROM {table}
        WHERE search_tsv @@ plainto_tsquery('spanish', %s)
        ORDER BY keyword_score DESC
        LIMIT %s;
        """,
        (query, query, limit),
    )
    rows = cursor.fetchall()
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def hybrid_fuse(
    vector_hits: List[Dict[str, object]],
    keyword_hits: List[Dict[str, object]],
    vector_weight: float,
    keyword_weight: float,
    rrf_k: int,
) -> List[Dict[str, object]]:
    by_chunk: Dict[str, Dict[str, object]] = {}

    for rank, item in enumerate(vector_hits, start=1):
        chunk_id = str(item["chunk_id"])
        candidate = by_chunk.setdefault(chunk_id, dict(item))
        candidate["vector_rank"] = rank
        candidate["vector_rrf"] = vector_weight * (1.0 / (rrf_k + rank))

    for rank, item in enumerate(keyword_hits, start=1):
        chunk_id = str(item["chunk_id"])
        candidate = by_chunk.setdefault(chunk_id, dict(item))
        candidate["keyword_rank"] = rank
        candidate["keyword_rrf"] = keyword_weight * (1.0 / (rrf_k + rank))

    merged: List[Dict[str, object]] = []
    for candidate in by_chunk.values():
        vector_rrf = float(candidate.get("vector_rrf", 0.0))
        keyword_rrf = float(candidate.get("keyword_rrf", 0.0))
        candidate["hybrid_score"] = vector_rrf + keyword_rrf
        merged.append(candidate)

    merged.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
    return merged


def apply_quality_prior(item: Dict[str, object]) -> float:
    ocr_quality = str(item.get("ocr_quality", "")).strip().lower()
    if ocr_quality == "good":
        return 0.05
    if ocr_quality == "usable_with_caution":
        return -0.02
    if ocr_quality in {"poor", "failed", "unknown"}:
        return -0.10
    return 0.0


def format_citation(item: Dict[str, object]) -> str:
    doc_id = str(item.get("doc_id", ""))
    page_start = int(item.get("page_start", 0) or 0)
    page_end = int(item.get("page_end", 0) or 0)
    if page_start and page_end and page_start != page_end:
        return f"[{doc_id} pp.{page_start}-{page_end}]"
    if page_start:
        return f"[{doc_id} p.{page_start}]"
    return f"[{doc_id}]"


def build_generator_payload(query: str, selected: Sequence[Dict[str, object]]) -> str:
    lines = [
        "### Pregunta del usuario",
        query,
        "",
        "### Evidencia recuperada (usar solo estos fragmentos)",
    ]
    for idx, item in enumerate(selected, start=1):
        citation = format_citation(item)
        title = str(item.get("title", "")).strip()
        section = str(item.get("section_title", "")).strip()
        text = str(item.get("text", "")).strip()
        lines.append(f"#### Chunk {idx} {citation}")
        lines.append(f"Título: {title}")
        if section:
            lines.append(f"Sección: {section}")
        lines.append(text)
        lines.append("")
    lines.append("### Reglas")
    lines.append("- Responde solo con base en la evidencia anterior.")
    lines.append("- Cita siempre usando el formato [DOC_ID p.X] o [DOC_ID pp.X-Y].")
    lines.append("- Si la evidencia es insuficiente, indícalo explícitamente.")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    table = validate_table_name(args.table)

    if not args.dsn:
        raise SystemExit("Missing --dsn and DATABASE_URL is not set.")
    if not args.query.strip():
        raise SystemExit("Query cannot be empty.")

    normalized_query = normalize_query(args.query)

    try:
        import psycopg  # type: ignore
    except Exception as exc:
        raise SystemExit("Install psycopg first: pip install 'psycopg[binary]'") from exc

    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore
    except Exception as exc:
        raise SystemExit("Install sentence-transformers first: pip install sentence-transformers") from exc

    embedder = SentenceTransformer(args.embedding_model)
    query_vector = embedder.encode([normalized_query], normalize_embeddings=True)[0]
    query_vector_literal = to_vector_literal([float(value) for value in query_vector])

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cursor:
            vector_hits = fetch_vector_hits(
                cursor=cursor,
                table=table,
                query_vector_literal=query_vector_literal,
                limit=max(args.k_vector, 1),
            )
            keyword_hits = fetch_keyword_hits(
                cursor=cursor,
                table=table,
                query=normalized_query,
                limit=max(args.k_keyword, 1),
            )

    merged = hybrid_fuse(
        vector_hits=vector_hits,
        keyword_hits=keyword_hits,
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight,
        rrf_k=max(args.rrf_k, 1),
    )
    merged = merged[: max(args.k_merge, 1)]

    if not merged:
        print(json.dumps({"query": args.query, "results": []}, ensure_ascii=False, indent=2))
        return 0

    reranker = CrossEncoder(args.reranker_model)
    rerank_pairs = [(normalized_query, str(item.get("text", ""))) for item in merged]
    rerank_scores = reranker.predict(rerank_pairs)

    for item, rerank_score in zip(merged, rerank_scores):
        hybrid_score = float(item.get("hybrid_score", 0.0))
        quality_prior = apply_quality_prior(item)
        item["rerank_score"] = float(rerank_score)
        item["quality_prior"] = quality_prior
        item["final_score"] = float(rerank_score) + (0.2 * hybrid_score) + quality_prior
        item["citation"] = format_citation(item)

    merged.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    final_results = merged[: max(args.k_final, 1)]

    response = {
        "query": args.query,
        "normalized_query": normalized_query,
        "retrieval": {
            "vector_hits": len(vector_hits),
            "keyword_hits": len(keyword_hits),
            "merged_candidates": len(merged),
            "returned_results": len(final_results),
            "table": table,
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
        },
        "results": final_results,
    }
    print(json.dumps(response, ensure_ascii=False, indent=2))

    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = build_generator_payload(query=args.query, selected=final_results)
        args.prompt_out.write_text(prompt_text, encoding="utf-8")
        print(f"Prompt bundle written to: {args.prompt_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
