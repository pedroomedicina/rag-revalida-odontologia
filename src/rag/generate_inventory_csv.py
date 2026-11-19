#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

PDF_PAGE_RE = re.compile(rb"/Type\s*/Page\b")


def iter_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext in extensions:
            yield path


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "document"


def title_from_filename(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or path.name


def estimate_pdf_page_count(path: Path) -> int:
    # Quick and dependency-free fallback page estimate.
    data = path.read_bytes()
    return len(PDF_PAGE_RE.findall(data))


def pdf_page_count(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        count = len(reader.pages)
        return str(count) if count > 0 else ""
    except Exception:
        count = estimate_pdf_page_count(path)
        return str(count) if count > 0 else ""


def build_doc_id(prefix: str, index: int, filename: str) -> str:
    return f"{prefix}-{index:04d}-{slugify(filename)[:32]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate corpus_inventory.csv from a cleaned corpus folder."
    )
    parser.add_argument("source", type=Path, help="Cleaned corpus directory")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/corpus_inventory.csv"),
        help="Output CSV path (default: ./data/processed/corpus_inventory.csv)",
    )
    parser.add_argument(
        "--extensions",
        default="pdf,doc,docx,docm,dot,dotx",
        help="Comma-separated extensions to include (default: pdf/doc/docx family).",
    )
    parser.add_argument(
        "--doc-id-prefix",
        default="UCH-ODO-ENDO",
        help="Prefix used for generated doc_id values.",
    )
    parser.add_argument("--course-program", default="ENDODONCIA")
    parser.add_argument("--faculty-career", default="ODONTOLOGIA")
    parser.add_argument("--year", default="")
    parser.add_argument("--semester", default="")
    parser.add_argument("--source-type", default="guia_apunte")
    parser.add_argument("--authority-level", default="media")
    parser.add_argument("--rights-status", default="open")
    parser.add_argument("--ocr-quality", default="unknown")
    parser.add_argument("--status", default="ready")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    extensions = {
        ext.strip().lower().lstrip(".") for ext in args.extensions.split(",") if ext.strip()
    }
    if not extensions:
        raise SystemExit("No valid extensions were provided.")

    files: list[Path] = sorted(iter_files(source, extensions), key=lambda p: p.name.lower())
    if not files:
        raise SystemExit(f"No files found in {source} for extensions: {sorted(extensions)}")

    fieldnames = [
        "doc_id",
        "filename",
        "source_path",
        "title",
        "course_program",
        "faculty_career",
        "year",
        "semester",
        "source_type",
        "authority_level",
        "rights_status",
        "page_count",
        "ocr_quality",
        "status",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for idx, path in enumerate(files, start=1):
            ext = path.suffix.lower().lstrip(".")
            page_count = ""
            if ext == "pdf":
                page_count = pdf_page_count(path)

            writer.writerow(
                {
                    "doc_id": build_doc_id(args.doc_id_prefix, idx, path.stem),
                    "filename": path.name,
                    "source_path": str(path),
                    "title": title_from_filename(path),
                    "course_program": args.course_program,
                    "faculty_career": args.faculty_career,
                    "year": args.year,
                    "semester": args.semester,
                    "source_type": args.source_type,
                    "authority_level": args.authority_level,
                    "rights_status": args.rights_status,
                    "page_count": page_count,
                    "ocr_quality": args.ocr_quality,
                    "status": args.status,
                }
            )

    print(f"Source: {source}")
    print(f"Files included: {len(files)}")
    print(f"Inventory CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
