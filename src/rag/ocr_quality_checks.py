#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
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


MIN_TEXT_CHARS = 20  # letters/digits; below this there is not enough text to judge
MIN_PAGE_COVERAGE = 0.8  # same threshold ingestion uses before trying OCR
GOOD_SCORE = 0.75
USABLE_SCORE = 0.45
CAUTION_CAP = 0.74
MAX_WORD_LENGTH = 25

ODD_SYMBOL_RE = re.compile(r"[^\w\s.,;:¿?¡!\-()/%°•–—“”«»\"']")
WORD_SPLIT_RE = re.compile(r"[\s\-/–—]+")
NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")
VOWEL_RE = re.compile(r"[aeiouyáéíóúü]", re.IGNORECASE)
SINGLE_LETTER_WORDS = set("aeouyáéóú")


def plausible_word_ratio(text: str) -> float:
    """Share of words that look like words: only letters, with a vowel, not run together.

    Plain numbers are neutral. Catches OCR noise such as "h1dr0x1d0" or "E l c a l c i o".
    """
    words = plausible = 0
    for token in WORD_SPLIT_RE.split(text):
        token = token.strip(".,;:¿?¡!()[]%°•“”«»\"'")
        if not any(ch.isalnum() for ch in token) or NUMBER_RE.fullmatch(token):
            continue
        words += 1
        if len(token) == 1:
            plausible += token.lower() in SINGLE_LETTER_WORDS
        elif token.isalpha() and len(token) <= MAX_WORD_LENGTH and VOWEL_RE.search(token):
            plausible += 1
    return safe_ratio(plausible, words)


def label_for_score(score: float) -> str:
    if score >= GOOD_SCORE:
        return "good"
    if score >= USABLE_SCORE:
        return "usable_with_caution"
    return "poor" if score > 0 else "failed"


def score_text_quality(text: str) -> tuple[float, str]:
    """Score one page on character and word signals. Length is only a floor."""
    text = unicodedata.normalize("NFC", text or "")
    visible = [ch for ch in text if not ch.isspace()]
    total = len(visible)
    if total == 0:
        return 0.0, "failed"

    printable_ratio = safe_ratio(sum(ch.isprintable() for ch in visible), total)
    alpha_ratio = safe_ratio(sum(ch.isalpha() for ch in visible), total)
    odd_ratio = safe_ratio(len(ODD_SYMBOL_RE.findall(text)), total)

    score = (
        # Full marks from 95% plausible words, none at 50% or less.
        max(0.0, min(1.0, (plausible_word_ratio(text) - 0.5) / 0.45)) * 0.6
        + min(1.0, alpha_ratio / 0.7) * 0.2
        + printable_ratio * 0.1
        + max(0.0, 1.0 - odd_ratio * 4.0) * 0.1
    )
    return score, label_for_score(score)


def score_pages_quality(pages: list[str]) -> tuple[float, str]:
    """Length-weighted mean of page scores.

    Pages without text do not dilute the score, but a document missing text on too many
    pages cannot be "good", and one with almost no text at all is "poor".
    """
    texts = [page for page in pages if page.strip()]
    if not texts:
        return 0.0, "failed"
    weights = [sum(1 for ch in page if not ch.isspace()) for page in texts]
    scores = [score_text_quality(page)[0] for page in texts]
    score = sum(s * w for s, w in zip(scores, weights, strict=True)) / sum(weights)
    if safe_ratio(len(texts), len(pages)) < MIN_PAGE_COVERAGE:
        score = min(score, CAUTION_CAP)
    if sum(ch.isalnum() for page in texts for ch in page) < MIN_TEXT_CHARS:
        score = min(score, USABLE_SCORE - 0.01)
    return score, label_for_score(score)


def extract_pdf_text_sample(path: Path, max_pages: int = 3) -> list[str]:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages[:max_pages]]
    except Exception:
        return []


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


def read_artifact_pages(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
            texts = []
            for page in payload["pages"]:
                if isinstance(page, dict):
                    texts.append(str(page.get("text", "")))
            return texts
        return [json.dumps(payload, ensure_ascii=False)]
    return [path.read_text(encoding="utf-8", errors="ignore")]


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
        score, label = score_pages_quality(sample)
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

        pages = read_artifact_pages(artifact)
        score, label = score_pages_quality(pages)
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
