#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*)?[\)\.\-]?\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\s\-/]{4,}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest corpus files into normalized, page-aware JSON artifacts."
    )
    parser.add_argument("inventory_csv", type=Path, help="Path to corpus_inventory.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/normalized"),
        help="Directory to write normalized document JSON files",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/reports/ingestion_report.json"),
        help="Path to write ingestion report JSON",
    )
    parser.add_argument(
        "--only-status",
        default="ready",
        help="Comma-separated inventory status values to ingest (default: ready)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip docs whose normalized JSON already exists",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit with non-zero code if any document fails ingestion",
    )
    return parser


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def guess_sections(text: str) -> list[str]:
    sections: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if len(candidate) > 120:
            continue
        if HEADING_RE.match(candidate):
            sections.append(candidate)
        if len(sections) >= 8:
            break
    return sections


def sanitize_text(text: str) -> str:
    # Guard against invalid surrogate characters from malformed PDFs.
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def extract_pdf_pages_pypdf(path: Path) -> list[dict[str, object]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install with: pip3 install pypdf"
        ) from exc

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path), strict=False)
    pages: list[dict[str, object]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = sanitize_text(page.extract_text() or "")
        pages.append(
            {
                "page": idx,
                "text": text,
                "sections": guess_sections(text),
            }
        )
    return pages


def repair_pdf_with_pikepdf(input_pdf: Path, output_pdf: Path) -> bool:
    try:
        import pikepdf  # type: ignore
    except Exception:
        return False

    try:
        with pikepdf.open(str(input_pdf)) as pdf:
            pdf.save(str(output_pdf))
        return True
    except Exception:
        return False


def repair_pdf_with_qpdf(input_pdf: Path, output_pdf: Path) -> bool:
    qpdf_bin = shutil.which("qpdf")
    if not qpdf_bin:
        return False

    cmd = [qpdf_bin, "--repair", str(input_pdf), str(output_pdf)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.returncode == 0 and output_pdf.exists()


def ocr_pdf_with_ocrmypdf(input_pdf: Path, output_pdf: Path) -> bool:
    ocrmypdf_bin = shutil.which("ocrmypdf")
    if not ocrmypdf_bin:
        return False

    cmd = [
        ocrmypdf_bin,
        "--skip-text",
        "--output-type",
        "pdf",
        str(input_pdf),
        str(output_pdf),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.returncode == 0 and output_pdf.exists()


# Share of pages that must have extractable text before a PDF counts as having a text layer.
# Below this (scans, image-only slides) the OCR fallback is attempted.
MIN_TEXT_COVERAGE = 0.8


def text_coverage(pages: list[dict[str, object]]) -> float:
    if not pages:
        return 0.0
    return sum(1 for page in pages if str(page.get("text", "")).strip()) / len(pages)


def extract_pdf_with_fallback(path: Path) -> tuple[list[dict[str, object]], str, list[str]]:
    """Extract page text, repairing unreadable PDFs and OCR-ing PDFs without a text layer.

    Order: pypdf -> (if unreadable) pikepdf repair -> qpdf repair -> (if unreadable or
    low text coverage) ocrmypdf. A readable PDF whose text stays below MIN_TEXT_COVERAGE
    after every available step is returned as `<method>_low_text`, so moment-2 OCR checks
    can label it and send it to review instead of it silently passing as "ok".
    """
    attempts: list[str] = []
    pages: list[dict[str, object]] | None = None
    extractor = ""

    def accept(candidate: list[dict[str, object]], method: str, label: str) -> bool:
        """Keep the candidate if it beats the current one; True when good enough to stop."""
        nonlocal pages, extractor
        coverage = text_coverage(candidate)
        if pages is None or coverage > text_coverage(pages):
            pages, extractor = candidate, method
        if coverage >= MIN_TEXT_COVERAGE:
            attempts.append(f"{label}:ok")
            return True
        attempts.append(f"{label}:low_text:coverage={coverage:.2f}")
        return False

    try:
        if accept(extract_pdf_pages_pypdf(path), "pypdf", "pypdf"):
            return pages, extractor, attempts
    except Exception as exc:
        attempts.append(f"pypdf:fail:{exc}")

    with tempfile.TemporaryDirectory(prefix="ingest_pdf_") as tmpdir:
        tmp = Path(tmpdir)

        # Repairs only help when pypdf could not parse the file at all.
        if pages is None:
            repairs = (
                ("pikepdf", repair_pdf_with_pikepdf, "pypdf_after_pikepdf_repair"),
                ("qpdf", repair_pdf_with_qpdf, "pypdf_after_qpdf_repair"),
            )
            for name, repair, method in repairs:
                repaired = tmp / f"{path.stem}.{name}.repaired.pdf"
                if not repair(path, repaired):
                    attempts.append(f"repair_{name}:unavailable_or_failed")
                    continue
                attempts.append(f"repair_{name}:ok")
                try:
                    if accept(extract_pdf_pages_pypdf(repaired), method, f"pypdf_on_{name}"):
                        return pages, extractor, attempts
                    break  # readable now, but no text layer: go to OCR
                except Exception as exc:
                    attempts.append(f"pypdf_on_{name}:fail:{exc}")

        ocr_pdf = tmp / f"{path.stem}.ocr.pdf"
        if ocr_pdf_with_ocrmypdf(path, ocr_pdf):
            attempts.append("ocr_ocrmypdf:ok")
            try:
                if accept(extract_pdf_pages_pypdf(ocr_pdf), "ocrmypdf_then_pypdf", "pypdf_on_ocr"):
                    return pages, extractor, attempts
            except Exception as exc:
                attempts.append(f"pypdf_on_ocr:fail:{exc}")
        else:
            attempts.append("ocr_ocrmypdf:unavailable_or_failed")

    if pages is not None:
        attempts.append("kept_low_text_extraction")
        return pages, f"{extractor}_low_text", attempts

    raise RuntimeError("PDF extraction failed after fallback chain: " + " | ".join(attempts))


LEGACY_WORD_EXTENSIONS = {"doc", "dot"}


def convert_legacy_word(path: Path, out_dir: Path) -> tuple[Path, str]:
    """Convert a binary .doc/.dot to .docx with LibreOffice, or macOS textutil as a fallback."""
    tried: list[str] = []
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        result = subprocess.run(
            [soffice, "--headless", "--convert-to", "docx", "--outdir", str(out_dir), str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        converted = out_dir / f"{path.stem}.docx"
        if result.returncode == 0 and converted.exists():
            return converted, "soffice"
        tried.append(f"soffice:exit={result.returncode}")
    else:
        tried.append("soffice:not_installed")

    textutil = shutil.which("textutil")
    if textutil:
        converted = out_dir / f"{path.stem}.textutil.docx"
        result = subprocess.run(
            [textutil, "-convert", "docx", str(path), "-output", str(converted)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0 and converted.exists():
            return converted, "textutil"
        tried.append(f"textutil:exit={result.returncode}")
    else:
        tried.append("textutil:not_installed")

    raise RuntimeError(
        f"Legacy Word file needs conversion to .docx and no converter succeeded "
        f"({', '.join(tried)}). Install LibreOffice or convert the file to .docx: {path}"
    )


def extract_word_pages(path: Path) -> tuple[list[dict[str, object]], str, list[str]]:
    ext = path.suffix.lower().lstrip(".")
    if ext not in LEGACY_WORD_EXTENSIONS:
        pages, extractor = extract_docx_pages(path)
        return pages, extractor, [f"{extractor}:ok"]

    with tempfile.TemporaryDirectory(prefix="ingest_doc_") as tmpdir:
        converted, converter = convert_legacy_word(path, Path(tmpdir))
        pages, extractor = extract_docx_pages(converted)
    method = f"{extractor}_after_{converter}"
    return pages, method, [f"convert_{converter}:ok", f"{extractor}:ok"]


def extract_docx_pages(path: Path) -> tuple[list[dict[str, object]], str]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            xml_data = zf.read("word/document.xml")
    except Exception as exc:
        raise RuntimeError(f"Unable to read DOCX file: {path}") from exc

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml_data)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        text_nodes = paragraph.findall(".//w:t", ns)
        text = "".join(node.text or "" for node in text_nodes).strip()
        if text:
            paragraphs.append(text)

    merged_text = sanitize_text("\n".join(paragraphs))
    pages = [
        {
            "page": 1,
            "text": merged_text,
            "sections": guess_sections(merged_text),
        }
    ]
    return pages, "docx-xml"


def ingest_document(row: dict[str, str]) -> dict[str, object]:
    doc_id = (row.get("doc_id") or "").strip()
    source_path_value = (row.get("source_path") or "").strip()
    title = (row.get("title") or "").strip()
    filename = (row.get("filename") or "").strip()
    status = (row.get("status") or "").strip()

    if not doc_id:
        raise RuntimeError("Missing doc_id in inventory row")
    if not source_path_value:
        raise RuntimeError(f"{doc_id}: missing source_path")

    source_path = Path(source_path_value).expanduser().resolve()
    if not source_path.exists():
        raise RuntimeError(f"{doc_id}: source_path does not exist: {source_path}")

    ext = source_path.suffix.lower().lstrip(".")
    method_trace: list[str] = []
    if ext == "pdf":
        pages, extractor, method_trace = extract_pdf_with_fallback(source_path)
    elif ext in {"docx", "doc", "docm", "dot", "dotx"}:
        pages, extractor, method_trace = extract_word_pages(source_path)
    else:
        raise RuntimeError(f"{doc_id}: unsupported extension '{ext}'")

    total_chars = sum(len(str(page.get("text", ""))) for page in pages)
    non_empty_pages = sum(1 for page in pages if str(page.get("text", "")).strip())
    text_coverage = (non_empty_pages / len(pages)) if pages else 0.0

    return {
        "doc_id": doc_id,
        "title": title,
        "filename": filename,
        "source_path": str(source_path),
        "ingestion_status": "ok",
        "inventory_status": status,
        "extractor": extractor,
        "extraction_method": extractor,
        "extraction_trace": method_trace,
        "ingested_at": now_iso(),
        "stats": {
            "page_count": len(pages),
            "non_empty_pages": non_empty_pages,
            "text_chars": total_chars,
            "text_coverage": round(text_coverage, 4),
        },
        "pages": pages,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    inventory_csv = args.inventory_csv.expanduser().resolve()
    if not inventory_csv.exists():
        raise SystemExit(f"Inventory CSV not found: {inventory_csv}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.report_json.expanduser().resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)

    target_status = {item.strip() for item in args.only_status.split(",") if item.strip()} or {
        "ready"
    }

    rows = read_inventory(inventory_csv)
    report: dict[str, object] = {
        "inventory_csv": str(inventory_csv),
        "output_dir": str(output_dir),
        "ingested_at": now_iso(),
        "filters": {
            "only_status": sorted(target_status),
            "skip_existing": bool(args.skip_existing),
        },
        "totals": {
            "rows": len(rows),
            "selected": 0,
            "ingested_ok": 0,
            "skipped_existing": 0,
            "skipped_status": 0,
            "failed": 0,
        },
        "failures": [],
        "outputs_preview": [],
    }

    for row in rows:
        doc_id = (row.get("doc_id") or "").strip()
        status = (row.get("status") or "").strip()
        if status not in target_status:
            report["totals"]["skipped_status"] += 1
            continue
        report["totals"]["selected"] += 1

        if not doc_id:
            report["totals"]["failed"] += 1
            report["failures"].append({"doc_id": "", "error": "missing doc_id"})
            continue

        out_file = output_dir / f"{doc_id}.json"
        if args.skip_existing and out_file.exists():
            if out_file.stat().st_size > 0:
                report["totals"]["skipped_existing"] += 1
                continue

        try:
            normalized = ingest_document(row)
            out_file.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["totals"]["ingested_ok"] += 1
            if len(report["outputs_preview"]) < 30:
                report["outputs_preview"].append(str(out_file))
        except Exception as exc:
            report["totals"]["failed"] += 1
            report["failures"].append({"doc_id": doc_id, "error": str(exc)})

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    totals = report["totals"]
    print(f"Inventory rows: {totals['rows']}")
    print(f"Selected for ingestion: {totals['selected']}")
    print(f"Ingested OK: {totals['ingested_ok']}")
    print(f"Skipped (status): {totals['skipped_status']}")
    print(f"Skipped (existing): {totals['skipped_existing']}")
    print(f"Failed: {totals['failed']}")
    print(f"Output dir: {output_dir}")
    print(f"Report: {report_json}")

    if totals["failed"] > 0 and args.fail_on_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
