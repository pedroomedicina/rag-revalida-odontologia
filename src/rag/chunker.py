#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.ingest import HEADING_RE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create page-aware chunks from normalized document artifacts."
    )
    parser.add_argument("inventory_csv", type=Path, help="Path to corpus_inventory.csv")
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/normalized"),
        help="Directory containing normalized docs (*.json)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Output chunks JSONL path",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/reports/chunking_report.json"),
        help="Chunking report JSON path",
    )
    parser.add_argument(
        "--only-status",
        default="ready",
        help="Comma-separated inventory status values to chunk",
    )
    parser.add_argument(
        "--allowed-ocr-quality",
        default="good,usable_with_caution",
        help="Comma-separated allowed OCR quality values",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=300,
        help="Preferred minimum token count per chunk",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum token count per chunk",
    )
    return parser


# Line-level text structure. pypdf output has no blank lines between paragraphs, so a
# paragraph ends at a line that ends a sentence, at a list item, or at a heading.
SENTENCE_END_RE = re.compile(r"[.!?:;»\")]$")
LIST_ITEM_RE = re.compile(r"^(?:[-•●▪*·]\s|\d+[.)]\s|[a-z]\)\s|paso\s+\d+\b)", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MAX_HEADING_CHARS = 120


@dataclass(frozen=True)
class Block:
    text: str
    is_heading: bool


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _join_lines(lines: list[str]) -> str:
    joined = ""
    for line in lines:
        if joined.endswith("-") and line[:1].islower():
            joined += line  # "químico-" + "mecánica": keep the hyphen, drop the break
        elif joined:
            joined += " " + line
        else:
            joined = line
    return re.sub(r"\s+", " ", joined).strip()


def is_heading(line: str) -> bool:
    return len(line) <= MAX_HEADING_CHARS and bool(HEADING_RE.match(line))


def split_blocks(text: str, line_is_paragraph: bool = False) -> list[Block]:
    """Split page text into heading and paragraph blocks.

    With line_is_paragraph (DOCX extraction), every line is already a paragraph.
    """
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            joined = _join_lines(buffer)
            if joined:
                blocks.append(Block(joined, is_heading=False))
            buffer.clear()

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if is_heading(line):
            flush()
            blocks.append(Block(re.sub(r"\s+", " ", line), is_heading=True))
            continue
        if LIST_ITEM_RE.match(line):
            flush()
        buffer.append(line)
        if line_is_paragraph or SENTENCE_END_RE.search(line):
            flush()
    flush()
    return blocks


def split_paragraphs(text: str) -> list[str]:
    return [block.text for block in split_blocks(text)]


def split_to_limits(text: str, first_limit: int, limit: int) -> list[str]:
    """Split text into pieces of at most first_limit, then limit, tokens.

    Breaks at sentence boundaries; a single sentence longer than the limit is cut by words.
    """
    first_limit = max(first_limit, 1)
    if estimate_tokens(text) <= first_limit:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def budget() -> int:
        return first_limit if not pieces else limit

    def emit() -> None:
        nonlocal current, current_tokens
        if current:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0

    for sentence in SENTENCE_SPLIT_RE.split(text):
        words = sentence.split()
        if not words:
            continue
        if current_tokens + len(words) > budget():
            emit()
        while len(words) > budget():
            cut = budget()
            pieces.append(" ".join(words[:cut]))
            words = words[cut:]
        if words:
            current.append(" ".join(words))
            current_tokens += len(words)
    emit()
    return pieces


def read_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("doc_id", "")).strip(): row for row in reader}


def finalize_chunk(
    chunks: list[dict[str, object]],
    doc_meta: dict[str, str],
    doc_json: dict[str, object],
    text_parts: list[str],
    page_start: int,
    page_end: int,
    section_title: str,
) -> None:
    text = "\n\n".join(text_parts).strip()
    if not text:
        return
    idx = len(chunks) + 1
    doc_id = str(doc_meta.get("doc_id", "")).strip()
    chunk_id = f"{doc_id}-ch-{idx:04d}"
    chunks.append(
        {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "title": doc_meta.get("title", ""),
            "filename": doc_meta.get("filename", ""),
            "page_start": page_start,
            "page_end": page_end,
            "section_title": section_title or "",
            "content_type": "conceptual",
            "ocr_quality": doc_meta.get("ocr_quality", ""),
            "extraction_method": doc_json.get("extraction_method", doc_json.get("extractor", "")),
            "token_estimate": estimate_tokens(text),
            "text": text,
        }
    )


def chunk_document(
    doc_meta: dict[str, str],
    doc_json: dict[str, object],
    min_tokens: int,
    max_tokens: int,
) -> list[dict[str, object]]:
    """Group heading/paragraph blocks into chunks of at most max_tokens.

    - A heading always starts a new chunk and becomes its section_title; the section
      carries over to following pages until the next heading.
    - A chunk continues onto the next page when pages are contiguous, so steps that
      straddle a page break stay together. An empty page in between ends the chunk.
    - Paragraphs longer than max_tokens are split at sentence boundaries.
    - min_tokens is a soft target: chunks only come out shorter at a heading, a page
      gap, the end of the document, or before a paragraph that would overflow max_tokens.
    """
    del min_tokens  # see docstring; kept for CLI and caller compatibility
    pages = doc_json.get("pages", [])
    if not isinstance(pages, list):
        return []
    line_is_paragraph = str(doc_json.get("extraction_method", "")).startswith("docx")

    chunks: list[dict[str, object]] = []
    parts: list[str] = []
    tokens = 0
    page_start = page_end = 0
    has_body = False
    section = ""
    chunk_section = ""

    def add(text: str, page_num: int, body: bool) -> None:
        nonlocal tokens, page_start, page_end, has_body, chunk_section
        if not parts:
            page_start = page_num
            chunk_section = section
        parts.append(text)
        tokens += estimate_tokens(text)
        page_end = page_num
        has_body = has_body or body

    def flush() -> None:
        nonlocal parts, tokens, page_start, page_end, has_body
        if has_body:
            finalize_chunk(
                chunks=chunks,
                doc_meta=doc_meta,
                doc_json=doc_json,
                text_parts=parts,
                page_start=page_start,
                page_end=page_end,
                section_title=chunk_section,
            )
        parts, tokens, page_start, page_end, has_body = [], 0, 0, 0, False

    for page_obj in pages:
        if not isinstance(page_obj, dict):
            continue
        page_num = int(page_obj.get("page", 0) or 0)
        blocks = split_blocks(str(page_obj.get("text", "") or ""), line_is_paragraph)
        if not blocks:
            continue
        if parts and page_num != page_end + 1:
            flush()  # never merge across a gap in the page sequence

        for block in blocks:
            if block.is_heading:
                if has_body:
                    flush()
                section = block.text
                if not has_body:
                    chunk_section = section
                add(block.text, page_num, body=False)
                continue

            pending_heading_tokens = 0 if has_body else tokens
            pieces = split_to_limits(block.text, max_tokens - pending_heading_tokens, max_tokens)
            for piece in pieces:
                if has_body and tokens + estimate_tokens(piece) > max_tokens:
                    flush()
                add(piece, page_num, body=True)

    flush()
    return chunks


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    inventory_csv = args.inventory_csv.expanduser().resolve()
    normalized_dir = args.normalized_dir.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    report_json = args.report_json.expanduser().resolve()

    if not inventory_csv.exists():
        raise SystemExit(f"Inventory CSV not found: {inventory_csv}")
    if not normalized_dir.exists() or not normalized_dir.is_dir():
        raise SystemExit(f"Normalized docs directory not found: {normalized_dir}")
    if args.min_tokens < 1 or args.max_tokens < 1 or args.min_tokens > args.max_tokens:
        raise SystemExit("Invalid token thresholds: ensure 1 <= min_tokens <= max_tokens")

    allowed_status = {s.strip() for s in args.only_status.split(",") if s.strip()}
    allowed_ocr = {q.strip() for q in args.allowed_ocr_quality.split(",") if q.strip()}

    inventory = read_inventory(inventory_csv)
    doc_files = sorted(normalized_dir.glob("*.json"), key=lambda p: p.name.lower())

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_docs_chunked = 0
    skipped_missing_inventory = 0
    skipped_status = 0
    skipped_ocr = 0
    failures: list[dict[str, str]] = []
    doc_chunk_counts: dict[str, int] = {}

    with output_jsonl.open("w", encoding="utf-8") as out_handle:
        for doc_file in doc_files:
            doc_id = doc_file.stem
            meta = inventory.get(doc_id)
            if not meta:
                skipped_missing_inventory += 1
                continue

            status = str(meta.get("status", "")).strip()
            if status not in allowed_status:
                skipped_status += 1
                continue

            ocr_quality = str(meta.get("ocr_quality", "")).strip()
            if ocr_quality not in allowed_ocr:
                skipped_ocr += 1
                continue

            try:
                doc_json = json.loads(doc_file.read_text(encoding="utf-8"))
                chunks = chunk_document(
                    doc_meta=meta,
                    doc_json=doc_json,
                    min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                )
                if chunks:
                    total_docs_chunked += 1
                    doc_chunk_counts[doc_id] = len(chunks)
                for chunk in chunks:
                    out_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1
            except Exception as exc:
                failures.append({"doc_id": doc_id, "error": str(exc)})

    report = {
        "inventory_csv": str(inventory_csv),
        "normalized_dir": str(normalized_dir),
        "output_jsonl": str(output_jsonl),
        "thresholds": {
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
        },
        "filters": {
            "only_status": sorted(allowed_status),
            "allowed_ocr_quality": sorted(allowed_ocr),
        },
        "totals": {
            "normalized_docs": len(doc_files),
            "docs_chunked": total_docs_chunked,
            "chunks_written": total_chunks,
            "skipped_missing_inventory": skipped_missing_inventory,
            "skipped_status": skipped_status,
            "skipped_ocr": skipped_ocr,
            "failed_docs": len(failures),
        },
        "failures": failures[:50],
        "doc_chunk_counts_preview": dict(list(sorted(doc_chunk_counts.items()))[:40]),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Normalized docs: {len(doc_files)}")
    print(f"Docs chunked: {total_docs_chunked}")
    print(f"Chunks written: {total_chunks}")
    print(f"Skipped by status: {skipped_status}")
    print(f"Skipped by OCR quality: {skipped_ocr}")
    print(f"Failures: {len(failures)}")
    print(f"Chunks output: {output_jsonl}")
    print(f"Chunking report: {report_json}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
