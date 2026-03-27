#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import tempfile
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_inventory(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def guess_sections(text: str) -> List[str]:
    sections: List[str] = []
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


def extract_pdf_pages_pypdf(path: Path) -> List[Dict[str, object]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install with: pip3 install pypdf"
        ) from exc

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path), strict=False)
    pages: List[Dict[str, object]] = []
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


def extract_pdf_with_fallback(path: Path) -> Tuple[List[Dict[str, object]], str, List[str]]:
    attempts: List[str] = []

    try:
        pages = extract_pdf_pages_pypdf(path)
        attempts.append("pypdf:ok")
        return pages, "pypdf", attempts
    except Exception as exc:
        attempts.append(f"pypdf:fail:{exc}")

    with tempfile.TemporaryDirectory(prefix="ingest_pdf_") as tmpdir:
        tmp = Path(tmpdir)
        repaired_pikepdf = tmp / f"{path.stem}.pikepdf.repaired.pdf"
        repaired_qpdf = tmp / f"{path.stem}.qpdf.repaired.pdf"
        ocr_pdf = tmp / f"{path.stem}.ocr.pdf"

        if repair_pdf_with_pikepdf(path, repaired_pikepdf):
            try:
                pages = extract_pdf_pages_pypdf(repaired_pikepdf)
                attempts.append("repair_pikepdf:ok")
                attempts.append("pypdf_on_pikepdf:ok")
                return pages, "pypdf_after_pikepdf_repair", attempts
            except Exception as exc:
                attempts.append("repair_pikepdf:ok")
                attempts.append(f"pypdf_on_pikepdf:fail:{exc}")
        else:
            attempts.append("repair_pikepdf:unavailable_or_failed")

        if repair_pdf_with_qpdf(path, repaired_qpdf):
            try:
                pages = extract_pdf_pages_pypdf(repaired_qpdf)
                attempts.append("repair_qpdf:ok")
                attempts.append("pypdf_on_qpdf:ok")
                return pages, "pypdf_after_qpdf_repair", attempts
            except Exception as exc:
                attempts.append("repair_qpdf:ok")
                attempts.append(f"pypdf_on_qpdf:fail:{exc}")
        else:
            attempts.append("repair_qpdf:unavailable_or_failed")

        if ocr_pdf_with_ocrmypdf(path, ocr_pdf):
            try:
                pages = extract_pdf_pages_pypdf(ocr_pdf)
                attempts.append("ocr_ocrmypdf:ok")
                attempts.append("pypdf_on_ocr_pdf:ok")
                return pages, "ocrmypdf_then_pypdf", attempts
            except Exception as exc:
                attempts.append("ocr_ocrmypdf:ok")
                attempts.append(f"pypdf_on_ocr_pdf:fail:{exc}")
        else:
            attempts.append("ocr_ocrmypdf:unavailable_or_failed")

    raise RuntimeError("PDF extraction failed after fallback chain: " + " | ".join(attempts))


def extract_docx_pages(path: Path) -> Tuple[List[Dict[str, object]], str]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            xml_data = zf.read("word/document.xml")
    except Exception as exc:
        raise RuntimeError(f"Unable to read DOCX file: {path}") from exc

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml_data)
    paragraphs: List[str] = []
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


def ingest_document(row: Dict[str, str]) -> Dict[str, object]:
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
    method_trace: List[str] = []
    if ext == "pdf":
        pages, extractor, method_trace = extract_pdf_with_fallback(source_path)
    elif ext in {"docx", "doc", "docm", "dot", "dotx"}:
        pages, extractor = extract_docx_pages(source_path)
        method_trace = [f"{extractor}:ok"]
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

    target_status = {
        item.strip() for item in args.only_status.split(",") if item.strip()
    } or {"ready"}

    rows = read_inventory(inventory_csv)
    report: Dict[str, object] = {
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
