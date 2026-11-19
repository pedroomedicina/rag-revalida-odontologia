"""Ingestion fallback chain, with external tools (ocrmypdf, soffice, textutil) faked."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag import ingest
from tests.fixtures.corpus_builder import GUIDE_PAGES, write_docx, write_pdf


@pytest.fixture
def no_external_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "ocr_pdf_with_ocrmypdf", lambda src, dst: False)
    monkeypatch.setattr(ingest, "repair_pdf_with_qpdf", lambda src, dst: False)
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)


def test_text_pdf_is_extracted_directly(tmp_path, no_external_tools):
    pdf = tmp_path / "guia.pdf"
    write_pdf(pdf, GUIDE_PAGES)
    pages, method, trace = ingest.extract_pdf_with_fallback(pdf)
    assert (method, trace) == ("pypdf", ["pypdf:ok"])
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "hidróxido de calcio" in pages[0]["text"]


def test_scan_without_ocr_is_kept_but_marked_low_text(tmp_path, no_external_tools):
    pdf = tmp_path / "scan.pdf"
    write_pdf(pdf, [[""], [""]], text_layer=False)
    pages, method, trace = ingest.extract_pdf_with_fallback(pdf)
    assert method == "pypdf_low_text"
    assert trace == [
        "pypdf:low_text:coverage=0.00",
        "ocr_ocrmypdf:unavailable_or_failed",
        "kept_low_text_extraction",
    ]
    assert len(pages) == 2


def test_scan_uses_ocr_output_when_available(tmp_path, monkeypatch, no_external_tools):
    scan = tmp_path / "scan.pdf"
    write_pdf(scan, [[""], [""], [""]], text_layer=False)
    ocr_result = tmp_path / "ocr_result.pdf"
    write_pdf(ocr_result, GUIDE_PAGES)

    def fake_ocr(src: Path, dst: Path) -> bool:
        shutil.copy(ocr_result, dst)
        return True

    monkeypatch.setattr(ingest, "ocr_pdf_with_ocrmypdf", fake_ocr)
    pages, method, trace = ingest.extract_pdf_with_fallback(scan)
    assert method == "ocrmypdf_then_pypdf"
    assert trace[-1] == "pypdf_on_ocr:ok"
    assert "EDTA" in pages[2]["text"]


def test_partially_scanned_pdf_triggers_ocr(tmp_path, no_external_tools):
    pdf = tmp_path / "mixed.pdf"
    write_pdf(pdf, [GUIDE_PAGES[0], [""], [""]])  # 1 of 3 pages has text
    _, method, trace = ingest.extract_pdf_with_fallback(pdf)
    assert trace[0] == "pypdf:low_text:coverage=0.33"
    assert method == "pypdf_low_text"


def test_unreadable_pdf_fails_after_full_chain(tmp_path, no_external_tools):
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"this is not a PDF file\n" * 20)
    with pytest.raises(RuntimeError, match="fallback chain") as excinfo:
        ingest.extract_pdf_with_fallback(pdf)
    message = str(excinfo.value)
    assert "pypdf:fail" in message and "ocr_ocrmypdf:unavailable_or_failed" in message


def test_docx_is_read_without_conversion(tmp_path, no_external_tools):
    docx = tmp_path / "apunte.docx"
    write_docx(docx, ["DIAGNÓSTICO PULPAR", "La biopulpectomía se realiza con pulpa vital."])
    pages, method, trace = ingest.extract_word_pages(docx)
    assert (method, trace) == ("docx-xml", ["docx-xml:ok"])
    assert "biopulpectomía" in pages[0]["text"]


def test_legacy_doc_without_converter_fails_clearly(tmp_path, no_external_tools):
    doc = tmp_path / "apunte.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    with pytest.raises(RuntimeError, match="Install LibreOffice or convert the file to .docx"):
        ingest.extract_word_pages(doc)


def test_legacy_doc_is_converted_when_a_converter_exists(tmp_path, monkeypatch):
    doc = tmp_path / "apunte.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100)

    def fake_convert(path: Path, out_dir: Path) -> tuple[Path, str]:
        converted = out_dir / "apunte.docx"
        write_docx(converted, ["Texto convertido."])
        return converted, "soffice"

    monkeypatch.setattr(ingest, "convert_legacy_word", fake_convert)
    pages, method, trace = ingest.extract_word_pages(doc)
    assert method == "docx-xml_after_soffice"
    assert trace == ["convert_soffice:ok", "docx-xml:ok"]
    assert pages[0]["text"] == "Texto convertido."
