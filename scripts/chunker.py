#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


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


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", text or "")
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return [p for p in cleaned if p]


def read_inventory(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("doc_id", "")).strip(): row for row in reader}


def finalize_chunk(
    chunks: List[Dict[str, object]],
    doc_meta: Dict[str, str],
    doc_json: Dict[str, object],
    text_parts: List[str],
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
    doc_meta: Dict[str, str],
    doc_json: Dict[str, object],
    min_tokens: int,
    max_tokens: int,
) -> List[Dict[str, object]]:
    pages = doc_json.get("pages", [])
    if not isinstance(pages, list):
        return []

    chunks: List[Dict[str, object]] = []
    current_parts: List[str] = []
    current_tokens = 0
    current_page_start = 0
    current_page_end = 0
    current_section = ""

    for page_obj in pages:
        if not isinstance(page_obj, dict):
            continue
        page_num = int(page_obj.get("page", 0) or 0)
        page_text = str(page_obj.get("text", "") or "")
        page_sections = page_obj.get("sections", [])
        page_section_title = ""
        if isinstance(page_sections, list) and page_sections:
            page_section_title = str(page_sections[0] or "").strip()

        paragraphs = split_paragraphs(page_text)
        if not paragraphs:
            continue

        for paragraph in paragraphs:
            paragraph_tokens = estimate_tokens(paragraph)
            if paragraph_tokens == 0:
                continue

            section_switch = bool(page_section_title and current_section and page_section_title != current_section)
            page_switch = bool(current_page_end and page_num != current_page_end)
            would_exceed = current_tokens + paragraph_tokens > max_tokens

            if current_parts and (section_switch or page_switch or (would_exceed and current_tokens >= min_tokens)):
                finalize_chunk(
                    chunks=chunks,
                    doc_meta=doc_meta,
                    doc_json=doc_json,
                    text_parts=current_parts,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    section_title=current_section,
                )
                current_parts = []
                current_tokens = 0
                current_page_start = 0
                current_page_end = 0
                current_section = ""

            if not current_parts:
                current_page_start = page_num
                current_section = page_section_title or current_section

            current_parts.append(paragraph)
            current_tokens += paragraph_tokens
            current_page_end = page_num
            if page_section_title and not current_section:
                current_section = page_section_title

    if current_parts:
        finalize_chunk(
            chunks=chunks,
            doc_meta=doc_meta,
            doc_json=doc_json,
            text_parts=current_parts,
            page_start=current_page_start,
            page_end=current_page_end,
            section_title=current_section,
        )

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
    failures: List[Dict[str, str]] = []
    doc_chunk_counts: Dict[str, int] = {}

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
