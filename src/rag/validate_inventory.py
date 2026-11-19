#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate corpus_inventory.csv against a metadata template."
    )
    parser.add_argument("inventory_csv", type=Path, help="Path to corpus inventory CSV")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/metadata_template.json"),
        help="Path to metadata template JSON",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional validation report output path",
    )
    parser.add_argument(
        "--enforce-ocr-ready",
        action="store_true",
        help="Fail if ocr_quality is blocked by template OCR policy",
    )
    return parser


def load_template(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Template not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_row(
    row: dict[str, str],
    row_idx: int,
    template: dict[str, object],
    enforce_ocr_ready: bool,
) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    required_non_empty = template.get("required_non_empty", [])
    allowed_values = template.get("allowed_values", {})
    ocr_policy = template.get("ocr_policy", {})

    for column in required_non_empty:
        if not row.get(column, "").strip():
            errors.append(f"row {row_idx}: empty required field '{column}'")

    for column, allowed in allowed_values.items():
        value = row.get(column, "").strip()
        if value and value not in allowed:
            errors.append(
                f"row {row_idx}: invalid value '{value}' for '{column}', allowed={allowed}"
            )

    source_path = row.get("source_path", "").strip()
    if source_path:
        path = Path(source_path).expanduser()
        if not path.exists():
            errors.append(f"row {row_idx}: source_path does not exist: {source_path}")

    filename = row.get("filename", "").strip().lower()
    page_count = row.get("page_count", "").strip()
    if filename.endswith(".pdf"):
        if not page_count:
            warnings.append(f"row {row_idx}: missing page_count for PDF")
        else:
            try:
                page_count_value = int(page_count)
                if page_count_value <= 0:
                    errors.append(f"row {row_idx}: non-positive page_count for PDF")
            except ValueError:
                errors.append(f"row {row_idx}: invalid integer page_count '{page_count}'")

    if enforce_ocr_ready:
        blocked = set(ocr_policy.get("blocked_before_chunking", []))
        ocr_quality = row.get("ocr_quality", "").strip()
        if ocr_quality in blocked:
            errors.append(f"row {row_idx}: ocr_quality '{ocr_quality}' blocks chunking readiness")

    return {"errors": errors, "warnings": warnings}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    inventory_csv = args.inventory_csv.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    template = load_template(template_path)

    if not inventory_csv.exists():
        raise SystemExit(f"Inventory CSV not found: {inventory_csv}")

    required_columns = template.get("required_columns", [])
    report = {
        "inventory_csv": str(inventory_csv),
        "template": str(template_path),
        "rows": 0,
        "errors": [],
        "warnings": [],
        "duplicate_doc_ids": [],
        "status_counts": {},
        "ocr_quality_counts": {},
    }

    with inventory_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        found_columns = reader.fieldnames or []

        missing_columns = [col for col in required_columns if col not in found_columns]
        if missing_columns:
            report["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")

        doc_id_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        ocr_counter: Counter[str] = Counter()

        for row_idx, row in enumerate(reader, start=2):
            report["rows"] += 1
            doc_id = (row.get("doc_id") or "").strip()
            if doc_id:
                doc_id_counter[doc_id] += 1
            status_counter[(row.get("status") or "").strip() or "<empty>"] += 1
            ocr_counter[(row.get("ocr_quality") or "").strip() or "<empty>"] += 1

            row_result = validate_row(
                row=row,
                row_idx=row_idx,
                template=template,
                enforce_ocr_ready=args.enforce_ocr_ready,
            )
            report["errors"].extend(row_result["errors"])
            report["warnings"].extend(row_result["warnings"])

    duplicate_doc_ids = [doc_id for doc_id, count in doc_id_counter.items() if count > 1]
    if duplicate_doc_ids:
        report["duplicate_doc_ids"] = duplicate_doc_ids
        report["errors"].append(
            f"Duplicate doc_id values: {', '.join(sorted(duplicate_doc_ids)[:20])}"
        )

    report["status_counts"] = dict(sorted(status_counter.items()))
    report["ocr_quality_counts"] = dict(sorted(ocr_counter.items()))

    if args.report_json:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Validation report: {report_path}")

    print(f"Inventory rows: {report['rows']}")
    print(f"Validation errors: {len(report['errors'])}")
    print(f"Validation warnings: {len(report['warnings'])}")
    print(f"Status counts: {report['status_counts']}")
    print(f"OCR quality counts: {report['ocr_quality_counts']}")

    if report["errors"]:
        print("Validation failed.")
        for error in report["errors"][:30]:
            print(f"ERROR: {error}")
        return 1

    print("Validation passed.")
    for warning in report["warnings"][:20]:
        print(f"WARN: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
