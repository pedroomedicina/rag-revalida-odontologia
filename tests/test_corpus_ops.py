"""Cleanup, inventory, OCR scoring, and validation helpers."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rag.corpus_cleanup import normalized_title_key, safe_normalized_filename
from rag.generate_inventory_csv import build_doc_id, slugify
from rag.ocr_quality_checks import score_text_quality
from rag.validate_inventory import validate_row
from tests.fixtures.corpus_builder import write_docx

TEMPLATE = {
    "required_non_empty": ["doc_id", "title"],
    "allowed_values": {"status": ["ready", "needs_review", "excluded"]},
    "ocr_policy": {"blocked_before_chunking": ["poor", "failed", "unknown"]},
}


def test_safe_filename_strips_accents_and_spaces():
    assert safe_normalized_filename(Path("Guía Endodoncia - Medicación.PDF")) == (
        "guia_endodoncia_-_medicacion.pdf"
    )


def test_title_key_ignores_numbered_copy_suffix():
    assert normalized_title_key(Path("Guía (1).pdf")) == normalized_title_key(Path("Guía.pdf"))


@pytest.mark.known_bug("'_copia' suffix is not stripped because \\b does not fire after '_'")
def test_title_key_ignores_copia_suffix():
    assert normalized_title_key(Path("guia_medicacion_copia.pdf")) == normalized_title_key(
        Path("guia_medicacion.pdf")
    )


def test_slugify_is_ascii():
    assert slugify("Medicación Intracanal (2024)") == "medicacion-intracanal-2024"


def test_doc_id_shape():
    assert re.fullmatch(r"UCH-guia-larga-[0-9a-f]{6}", build_doc_id("UCH", "Guía Larga.pdf"))


def test_doc_id_is_deterministic():
    assert build_doc_id("UCH", "guia.pdf") == build_doc_id("UCH", "guia.pdf")


def test_doc_id_distinguishes_same_name_in_different_folders():
    assert build_doc_id("UCH", "a/guia.pdf") != build_doc_id("UCH", "b/guia.pdf")


def test_doc_id_distinguishes_slugs_that_truncate_alike():
    long_a = "x" * 60 + "_parte_a.pdf"
    long_b = "x" * 60 + "_parte_b.pdf"
    assert build_doc_id("UCH", long_a) != build_doc_id("UCH", long_b)


def test_adding_a_file_keeps_existing_doc_ids(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("guia.docx", "obturacion.docx"):
        write_docx(corpus / name, ["texto"])

    def inventory_ids() -> dict[str, str]:
        out = tmp_path / "inventory.csv"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rag.generate_inventory_csv",
                str(corpus),
                "--output-csv",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        with out.open(encoding="utf-8", newline="") as handle:
            return {row["filename"]: row["doc_id"] for row in csv.DictReader(handle)}

    before = inventory_ids()
    write_docx(corpus / "acceso_cameral.docx", ["texto"])  # sorts first
    after = inventory_ids()
    assert len(after) == 3
    assert {name: after[name] for name in before} == before


def test_ocr_score_empty_text_fails():
    assert score_text_quality("") == (0.0, "failed")


def test_ocr_score_long_clean_text_is_good():
    _, label = score_text_quality("El hipoclorito de sodio disuelve tejido orgánico. " * 100)
    assert label == "good"


def test_ocr_score_garbage_is_not_good():
    _, label = score_text_quality("�#@%&*^~|<>{}[]" * 300)
    assert label != "good"


@pytest.mark.known_bug("score is 45% text length, so short clean text is never 'good'")
def test_ocr_score_short_clean_text_is_good():
    _, label = score_text_quality(
        "El hidróxido de calcio se indica como medicación intracanal entre sesiones."
    )
    assert label == "good"


def test_validate_row_reports_missing_and_invalid(tmp_path):
    result = validate_row(
        {"doc_id": "", "title": "x", "status": "maybe", "source_path": str(tmp_path / "nope")},
        2,
        TEMPLATE,
        enforce_ocr_ready=False,
    )
    errors = " ".join(result["errors"])
    assert "empty required field 'doc_id'" in errors
    assert "invalid value 'maybe'" in errors
    assert "source_path does not exist" in errors


def test_validate_row_ocr_gate(tmp_path):
    row = {"doc_id": "a", "title": "t", "status": "ready", "ocr_quality": "unknown"}
    assert validate_row(row, 2, TEMPLATE, enforce_ocr_ready=False)["errors"] == []
    blocked = validate_row(row, 2, TEMPLATE, enforce_ocr_ready=True)["errors"]
    assert blocked and "blocks chunking" in blocked[0]
