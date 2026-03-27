#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MVP corpus preparation pipeline end-to-end."
    )
    parser.add_argument("source", type=Path, help="Raw or cleaned corpus folder")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/metadata_template.json"),
        help="Metadata template JSON path",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=None,
        help="Output clean folder (default: <source>_clean)",
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=Path("data/processed/corpus_inventory.csv"),
        help="Output inventory CSV path",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("artifacts/reports/inventory_validation_report.json"),
        help="Validation JSON report path",
    )
    parser.add_argument(
        "--ocr-m1-report",
        type=Path,
        default=Path("artifacts/reports/inventory_ocr_m1_report.json"),
        help="Moment-1 OCR check report path",
    )
    parser.add_argument(
        "--ocr-m2-report",
        type=Path,
        default=Path("artifacts/reports/inventory_ocr_m2_report.json"),
        help="Moment-2 OCR check report path",
    )
    parser.add_argument(
        "--moment2-text-dir",
        type=Path,
        default=None,
        help="Directory with ingested text artifacts for moment-2 OCR checks",
    )
    parser.add_argument(
        "--normalized-output-dir",
        type=Path,
        default=Path("data/normalized"),
        help="Output directory for ingestion normalized documents",
    )
    parser.add_argument(
        "--ingestion-report",
        type=Path,
        default=Path("artifacts/reports/ingestion_report.json"),
        help="Ingestion report output path",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion step (only use if normalized docs already exist)",
    )
    parser.add_argument(
        "--strict-ingestion",
        action="store_true",
        help="Fail pipeline when any document ingestion fails",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunk generation step",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Chunk output JSONL path",
    )
    parser.add_argument(
        "--chunking-report",
        type=Path,
        default=Path("artifacts/reports/chunking_report.json"),
        help="Chunking report JSON path",
    )
    parser.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=300,
        help="Preferred minimum token estimate per chunk",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=800,
        help="Maximum token estimate per chunk",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup and use source directly for inventory",
    )
    parser.add_argument(
        "--enforce-ocr-ready",
        action="store_true",
        help="Fail validation if OCR quality blocks chunking readiness",
    )
    parser.add_argument(
        "--keep-extensions",
        default="pdf,doc,docx,docm,dot,dotx",
        help="Extension allowlist passed to cleanup/inventory scripts",
    )
    return parser


def load_template(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Template not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(cmd: List[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    template_path = args.template.expanduser().resolve()
    template = load_template(template_path)
    defaults = template.get("defaults", {})

    source = args.source.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    clean_output = (
        args.clean_output.expanduser().resolve()
        if args.clean_output
        else (source.parent / f"{source.name}_clean").resolve()
    )
    inventory_csv = args.inventory_csv.expanduser().resolve()
    validation_report = args.validation_report.expanduser().resolve()
    ocr_m1_report = args.ocr_m1_report.expanduser().resolve()
    ocr_m2_report = args.ocr_m2_report.expanduser().resolve()
    normalized_output_dir = args.normalized_output_dir.expanduser().resolve()
    ingestion_report = args.ingestion_report.expanduser().resolve()
    chunks_output = args.chunks_output.expanduser().resolve()
    chunking_report = args.chunking_report.expanduser().resolve()

    corpus_for_inventory = source
    if not args.skip_cleanup:
        cleanup_script = repo_root / "scripts" / "corpus_cleanup.py"
        cleanup_cmd = [
            sys.executable,
            str(cleanup_script),
            str(source),
            "--output",
            str(clean_output),
            "--apply",
            "--remove-junk",
            "--dedupe-hash",
            "--normalize-names",
            "--keep-extensions",
            args.keep_extensions,
            "--near-dup-pdf-resave-report",
        ]
        run_command(cleanup_cmd, "Step 1/7: Cleanup")
        corpus_for_inventory = clean_output
    else:
        print("\n=== Step 1/7: Cleanup skipped ===")
        print(f"Using source directly: {source}")

    generator_script = repo_root / "scripts" / "generate_inventory_csv.py"
    generate_cmd = [
        sys.executable,
        str(generator_script),
        str(corpus_for_inventory),
        "--output-csv",
        str(inventory_csv),
        "--extensions",
        args.keep_extensions,
        "--doc-id-prefix",
        str(defaults.get("doc_id_prefix", "DOC")),
        "--course-program",
        str(defaults.get("course_program", "")),
        "--faculty-career",
        str(defaults.get("faculty_career", "")),
        "--year",
        str(defaults.get("year", "")),
        "--semester",
        str(defaults.get("semester", "")),
        "--source-type",
        str(defaults.get("source_type", "")),
        "--authority-level",
        str(defaults.get("authority_level", "")),
        "--rights-status",
        str(defaults.get("rights_status", "")),
        "--ocr-quality",
        str(defaults.get("ocr_quality", "unknown")),
        "--status",
        str(defaults.get("status", "ready")),
    ]
    run_command(generate_cmd, "Step 2/7: Generate inventory")

    ocr_script = repo_root / "scripts" / "ocr_quality_checks.py"
    ocr_m1_cmd = [
        sys.executable,
        str(ocr_script),
        str(inventory_csv),
        "--moment",
        "moment1",
        "--update-inventory",
        "--report-json",
        str(ocr_m1_report),
    ]
    run_command(ocr_m1_cmd, "Step 3/7: OCR check moment 1 (inventory-time)")

    if not args.skip_ingestion:
        ingest_script = repo_root / "scripts" / "ingest.py"
        ingest_cmd = [
            sys.executable,
            str(ingest_script),
            str(inventory_csv),
            "--output-dir",
            str(normalized_output_dir),
            "--report-json",
            str(ingestion_report),
        ]
        if args.strict_ingestion:
            ingest_cmd.append("--fail-on-errors")
        run_command(ingest_cmd, "Step 4/7: Ingestion")
    else:
        print("\n=== Step 4/7: Ingestion skipped ===")
        print("Using existing normalized documents.")

    text_dir = None
    if args.moment2_text_dir:
        text_dir = args.moment2_text_dir.expanduser().resolve()
    elif not args.skip_ingestion:
        text_dir = normalized_output_dir

    if text_dir:
        ocr_m2_cmd = [
            sys.executable,
            str(ocr_script),
            str(inventory_csv),
            "--moment",
            "moment2",
            "--text-dir",
            str(text_dir),
            "--update-inventory",
            "--report-json",
            str(ocr_m2_report),
        ]
        run_command(ocr_m2_cmd, "Step 5/7: OCR check moment 2 (post-ingestion)")
    else:
        print("\n=== Step 5/7: OCR check moment 2 skipped ===")
        print(
            "Provide --moment2-text-dir or run ingestion so normalized text artifacts are available."
        )

    validator_script = repo_root / "scripts" / "validate_inventory.py"
    validate_cmd = [
        sys.executable,
        str(validator_script),
        str(inventory_csv),
        "--template",
        str(template_path),
        "--report-json",
        str(validation_report),
    ]
    if args.enforce_ocr_ready:
        validate_cmd.append("--enforce-ocr-ready")
    run_command(validate_cmd, "Step 6/7: Validate inventory")

    if not args.skip_chunking:
        chunker_script = repo_root / "scripts" / "chunker.py"
        chunk_cmd = [
            sys.executable,
            str(chunker_script),
            str(inventory_csv),
            "--normalized-dir",
            str(normalized_output_dir),
            "--output-jsonl",
            str(chunks_output),
            "--report-json",
            str(chunking_report),
            "--min-tokens",
            str(args.chunk_min_tokens),
            "--max-tokens",
            str(args.chunk_max_tokens),
        ]
        run_command(chunk_cmd, "Step 7/7: Chunking")
    else:
        print("\n=== Step 7/7: Chunking skipped ===")
        print("Use --skip-chunking only if chunks already exist.")

    print("\nPipeline completed successfully.")
    print(f"Inventory CSV: {inventory_csv}")
    print(f"OCR moment-1 report: {ocr_m1_report}")
    if not args.skip_ingestion:
        print(f"Ingestion report: {ingestion_report}")
        print(f"Normalized docs: {normalized_output_dir}")
    if text_dir:
        print(f"OCR moment-2 report: {ocr_m2_report}")
    print(f"Validation report: {validation_report}")
    if not args.skip_chunking:
        print(f"Chunks output: {chunks_output}")
        print(f"Chunking report: {chunking_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
