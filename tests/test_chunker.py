"""Chunker unit tests on hand-built normalized documents."""

from __future__ import annotations

import pytest

from rag.chunker import chunk_document, estimate_tokens, split_paragraphs

META = {"doc_id": "D1", "title": "Guía", "filename": "guia.pdf", "ocr_quality": "good"}


def doc(*pages: tuple[str, list[str]]) -> dict[str, object]:
    return {
        "extraction_method": "pypdf",
        "pages": [
            {"page": number, "text": text, "sections": sections}
            for number, (text, sections) in enumerate(pages, start=1)
        ],
    }


def words(n: int, word: str = "palabra") -> str:
    return " ".join([word] * n)


def test_split_paragraphs_on_blank_lines():
    assert split_paragraphs("uno\ndos\n\n  tres  \n\n\n") == ["uno dos", "tres"]


def test_estimate_tokens_counts_whitespace_tokens():
    assert estimate_tokens("a  b\nc") == 3


def test_chunks_carry_page_and_section_metadata():
    chunks = chunk_document(META, doc(("Intro\n\n" + words(10), ["INTRODUCCIÓN"])), 300, 800)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "D1-ch-0001"
    assert (chunk["page_start"], chunk["page_end"]) == (1, 1)
    assert chunk["section_title"] == "INTRODUCCIÓN"
    assert chunk["extraction_method"] == "pypdf"


def test_overflow_flushes_once_min_is_met():
    # 400 + 400 fits exactly at max; the third paragraph would overflow, so it starts a new chunk.
    text = "\n\n".join(words(400) for _ in range(3))
    chunks = chunk_document(META, doc((text, [])), 300, 800)
    assert [c["token_estimate"] for c in chunks] == [800, 400]


def test_empty_pages_are_skipped():
    chunks = chunk_document(META, doc(("", []), (words(5), [])), 300, 800)
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(2, 2)]


@pytest.mark.known_bug("one paragraph longer than max_tokens becomes one oversized chunk")
def test_long_paragraph_is_split_to_max_tokens():
    chunks = chunk_document(META, doc((words(1000), [])), 300, 800)
    assert all(c["token_estimate"] <= 800 for c in chunks)


@pytest.mark.known_bug("chunker always flushes at a page change, even for short contiguous pages")
def test_short_contiguous_pages_merge():
    chunks = chunk_document(META, doc((words(50), []), (words(50), [])), 300, 800)
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 2)]


@pytest.mark.known_bug("only the first heading on a page is used; mid-page headings are ignored")
def test_mid_page_heading_starts_new_chunk():
    text = f"IRRIGACIÓN\n\n{words(350)}\n\nPROTOCOLO FINAL\n\n{words(40)}"
    chunks = chunk_document(META, doc((text, ["IRRIGACIÓN", "PROTOCOLO FINAL"])), 300, 800)
    assert [c["section_title"] for c in chunks] == ["IRRIGACIÓN", "PROTOCOLO FINAL"]
