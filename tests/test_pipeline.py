"""End-to-end prep pipeline on the synthetic sample corpus (no database, no models)."""

from __future__ import annotations

import pytest


def test_pipeline_completes(prep_run):
    assert prep_run.returncode == 0, prep_run.stdout
    assert "Pipeline completed successfully." in prep_run.stdout


def test_cleanup_drops_junk_and_exact_duplicates(prep_run):
    cleanup = prep_run.cleanup_report
    assert cleanup["drops"]["junk_count"] == 1
    assert cleanup["dedupe"]["duplicate_extra_copies"] == 1
    assert cleanup["totals"]["copied_files"] == 6


def test_inventory_has_one_row_per_clean_file(prep_run):
    filenames = sorted(r["filename"] for r in prep_run.inventory)
    assert filenames == [
        "apunte_diagnostico.docx",
        "apunte_legacy.doc",
        "escaneo_clase.pdf",
        "guia_endodoncia_-_medicacion.pdf",
        "guia_endodoncia_medicacion_copia.pdf",
        "obturacion_larga.pdf",
    ]
    assert len({r["doc_id"] for r in prep_run.inventory}) == 6


def test_pdf_page_counts(prep_run):
    assert prep_run.row_for("guia_endodoncia_-_medicacion")["page_count"] == "3"
    assert prep_run.row_for("obturacion_larga")["page_count"] == "1"


def test_pdf_without_text_layer_is_flagged_for_review(prep_run):
    row = prep_run.row_for("escaneo_clase")
    assert row["ocr_quality"] == "failed"
    assert row["status"] == "needs_review"


def test_chunks_cite_real_pages(prep_run):
    pages = {r["doc_id"]: r["page_count"] for r in prep_run.inventory}
    assert prep_run.chunks, "no chunks written"
    for chunk in prep_run.chunks:
        assert 1 <= chunk["page_start"] <= chunk["page_end"]
        if pages[chunk["doc_id"]]:
            assert chunk["page_end"] <= int(pages[chunk["doc_id"]])
        assert chunk["chunk_id"].startswith(f"{chunk['doc_id']}-ch-")
        assert chunk["text"].strip()


def test_only_ready_documents_are_chunked(prep_run):
    ready = {
        r["doc_id"]
        for r in prep_run.inventory
        if r["status"] == "ready" and r["ocr_quality"] in {"good", "usable_with_caution"}
    }
    assert {c["doc_id"] for c in prep_run.chunks} <= ready


@pytest.mark.known_bug("ingestion failure leaves the document status=ready and exit code 0")
def test_failed_ingestion_is_not_left_ready(prep_run):
    failed = {f["doc_id"] for f in prep_run.report("ingestion")["failures"]}
    assert failed, "fixture should include a document that fails ingestion"
    for row in prep_run.inventory:
        if row["doc_id"] in failed:
            assert row["status"] != "ready"


@pytest.mark.known_bug("legacy .doc is parsed as a .docx zip and fails")
def test_legacy_doc_is_ingested_or_explicitly_unsupported(prep_run):
    failures = prep_run.report("ingestion")["failures"]
    assert not any("Unable to read DOCX" in f["error"] for f in failures)


@pytest.mark.known_bug("empty-text PDF counts as ingested OK, so the OCR fallback never runs")
def test_pdf_without_text_triggers_fallback(prep_run):
    doc = prep_run.normalized(prep_run.row_for("escaneo_clase")["doc_id"])
    assert doc["extraction_trace"] != ["pypdf:ok"]


@pytest.mark.known_bug("near-duplicate 'copia' resave is not detected")
def test_near_duplicate_resave_is_reported(prep_run):
    cleanup = prep_run.cleanup_report
    assert cleanup["near_duplicate_pdf_resave"]["candidate_pairs_count"] >= 1


@pytest.mark.known_bug("chunker flushes on every page, splitting procedures across pages")
def test_procedure_spanning_pages_stays_in_one_chunk(prep_run):
    doc_id = prep_run.row_for("guia_endodoncia_-_medicacion")["doc_id"]
    chunks = [c for c in prep_run.chunks if c["doc_id"] == doc_id]
    paso_1 = next(c for c in chunks if "Paso 1" in c["text"])
    assert "Paso 3" in paso_1["text"]


@pytest.mark.known_bug("a single long paragraph exceeds max_tokens")
def test_chunks_respect_max_tokens(prep_run):
    assert max(c["token_estimate"] for c in prep_run.chunks) <= 800
