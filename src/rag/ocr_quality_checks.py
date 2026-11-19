#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run split OCR quality checks (moment1 and moment2)."
    )
    parser.add_argument("inventory_csv", type=Path, help="Path to corpus inventory CSV")
    parser.add_argument(
        "--moment",
        choices=["moment1", "moment2"],
        required=True,
        help="Which OCR quality check moment to run",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=None,
        help="Directory containing ingested text artifacts (required for moment2)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--update-inventory",
        action="store_true",
        help="Persist OCR check results back into inventory CSV",
    )
    return parser


def read_inventory(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def write_inventory(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def score_text_quality(text: str) -> tuple[float, str]:
    cleaned = text or ""
    total = len(cleaned)
    if total == 0:
        return 0.0, "failed"

    printable = sum(1 for ch in cleaned if ch.isprintable())
    alpha = sum(1 for ch in cleaned if ch.isalpha())
    weird = len(re.findall(r"[^\w\s\.,;:¿\?¡!\-()/]", cleaned, flags=re.UNICODE))

    printable_ratio = safe_ratio(printable, total)
    alpha_ratio = safe_ratio(alpha, total)
    weird_ratio = safe_ratio(weird, total)

    score = (
        min(1.0, total / 4000.0) * 0.45
        + printable_ratio * 0.25
        + alpha_ratio * 0.2
        + max(0.0, 1.0 - weird_ratio * 4.0) * 0.1
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        label = "good"
    elif score >= 0.45:
        label = "usable_with_caution"
    elif score > 0:
        label = "poor"
    else:
        label = "failed"
    return score, label


def extract_pdf_text_sample(path: Path, max_pages: int = 3) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        chunks: list[str] = []
        for page in reader.pages[:max_pages]:
            page_text = page.extract_text() or ""
            if page_text:
                chunks.append(page_text)
        return "\n".join(chunks)
    except Exception:
        return ""


def detect_text_artifact(doc_id: str, source_path: str, text_dir: Path) -> Path | None:
    src_stem = Path(source_path).stem
    candidates = [
        text_dir / f"{doc_id}.txt",
        text_dir / f"{doc_id}.md",
        text_dir / f"{doc_id}.json",
        text_dir / f"{src_stem}.txt",
        text_dir / f"{src_stem}.md",
        text_dir / f"{src_stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_artifact_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
            texts = []
            for page in payload["pages"]:
                if isinstance(page, dict):
                    texts.append(str(page.get("text", "")))
            return "\n".join(texts)
        return json.dumps(payload, ensure_ascii=False)
    return path.read_text(encoding="utf-8", errors="ignore")


def ensure_headers(headers: list[str], additions: list[str]) -> list[str]:
    out = list(headers)
    for key in additions:
        if key not in out:
            out.append(key)
    return out


def run_moment1(rows: list[dict[str, str]]) -> dict[str, object]:
    summary = {"rows": len(rows), "checked": 0, "labels": {}, "issues": []}
    label_counter: dict[str, int] = {}

    for row in rows:
        source_path = (row.get("source_path") or "").strip()
        filename = (row.get("filename") or "").strip().lower()
        doc_id = (row.get("doc_id") or "").strip()
        row["ocr_m1_score"] = ""
        row["ocr_m1_label"] = "not_checked"
        row["ocr_m1_notes"] = ""

        if not source_path:
            row["ocr_m1_label"] = "missing_source"
            row["ocr_m1_notes"] = "missing source_path"
            summary["issues"].append(f"{doc_id}: missing source_path")
            label_counter[row["ocr_m1_label"]] = label_counter.get(row["ocr_m1_label"], 0) + 1
            continue

        path = Path(source_path)
        if not path.exists():
            row["ocr_m1_label"] = "missing_source"
            row["ocr_m1_notes"] = "source_path does not exist"
            summary["issues"].append(f"{doc_id}: source_path does not exist")
            label_counter[row["ocr_m1_label"]] = label_counter.get(row["ocr_m1_label"], 0) + 1
            continue

        if not filename.endswith(".pdf"):
            row["ocr_m1_label"] = "not_pdf"
            row["ocr_m1_notes"] = "moment1 heuristic skipped for non-PDF"
            label_counter[row["ocr_m1_label"]] = label_counter.get(row["ocr_m1_label"], 0) + 1
            continue

        sample = extract_pdf_text_sample(path)
        score, label = score_text_quality(sample)
        row["ocr_m1_score"] = f"{score:.3f}"
        row["ocr_m1_label"] = label
        row["ocr_m1_notes"] = "based on sample extracted text from first pages"
        summary["checked"] += 1
        label_counter[label] = label_counter.get(label, 0) + 1

    summary["labels"] = dict(sorted(label_counter.items()))
    return summary


def run_moment2(rows: list[dict[str, str]], text_dir: Path) -> dict[str, object]:
    summary = {"rows": len(rows), "checked": 0, "labels": {}, "issues": []}
    label_counter: dict[str, int] = {}

    for row in rows:
        doc_id = (row.get("doc_id") or "").strip()
        source_path = (row.get("source_path") or "").strip()
        row["ocr_m2_score"] = ""
        row["ocr_m2_notes"] = ""

        artifact = detect_text_artifact(doc_id, source_path, text_dir)
        if artifact is None:
            row["ocr_m2_notes"] = "text artifact not found"
            summary["issues"].append(f"{doc_id}: text artifact not found")
            # A document that was meant to be ingested but produced no text cannot be
            # chunked or cited; send it to review instead of leaving it "ready".
            # Rows already excluded or under review keep their status.
            if (row.get("status") or "").strip() == "ready":
                row["ocr_quality"] = "failed"
                row["status"] = "needs_review"
                label_counter["failed"] = label_counter.get("failed", 0) + 1
            continue

        text = read_artifact_text(artifact)
        score, label = score_text_quality(text)
        row["ocr_m2_score"] = f"{score:.3f}"
        row["ocr_m2_notes"] = f"artifact={artifact.name}"
        row["ocr_quality"] = label
        if label in {"poor", "failed"}:
            row["status"] = "needs_review"
        summary["checked"] += 1
        label_counter[label] = label_counter.get(label, 0) + 1

    summary["labels"] = dict(sorted(label_counter.items()))
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    inventory_csv = args.inventory_csv.expanduser().resolve()
    if not inventory_csv.exists():
        raise SystemExit(f"Inventory CSV not found: {inventory_csv}")

    headers, rows = read_inventory(inventory_csv)
    if not rows:
        raise SystemExit(f"No rows found in inventory: {inventory_csv}")

    if args.moment == "moment1":
        headers = ensure_headers(headers, ["ocr_m1_score", "ocr_m1_label", "ocr_m1_notes"])
        summary = run_moment1(rows)
    else:
        if args.text_dir is None:
            raise SystemExit("--text-dir is required for moment2")
        text_dir = args.text_dir.expanduser().resolve()
        if not text_dir.exists() or not text_dir.is_dir():
            raise SystemExit(f"Text directory not found: {text_dir}")
        headers = ensure_headers(headers, ["ocr_m2_score", "ocr_m2_notes", "ocr_quality", "status"])
        summary = run_moment2(rows, text_dir)

    if args.update_inventory:
        write_inventory(inventory_csv, headers, rows)
        print(f"Inventory updated: {inventory_csv}")

    report = {
        "inventory_csv": str(inventory_csv),
        "moment": args.moment,
        "summary": summary,
    }
    if args.moment == "moment2" and args.text_dir:
        report["text_dir"] = str(args.text_dir.expanduser().resolve())

    if args.report_json:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OCR report: {report_path}")

    print(f"Moment: {args.moment}")
    print(f"Rows: {summary['rows']}")
    print(f"Checked: {summary['checked']}")
    print(f"Labels: {summary['labels']}")
    print(f"Issues: {len(summary['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
