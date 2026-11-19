"""Chunker unit tests on hand-built normalized documents."""

from __future__ import annotations

from rag.chunker import (
    Block,
    chunk_document,
    estimate_tokens,
    split_blocks,
    split_paragraphs,
    split_to_limits,
)

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
    chunks = chunk_document(META, doc(("INTRODUCCIÓN\n" + words(10), ["INTRODUCCIÓN"])), 300, 800)
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


def test_long_paragraph_is_split_to_max_tokens():
    chunks = chunk_document(META, doc((words(1000), [])), 300, 800)
    assert all(c["token_estimate"] <= 800 for c in chunks)


def test_short_contiguous_pages_merge():
    chunks = chunk_document(META, doc((words(50), []), (words(50), [])), 300, 800)
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 2)]


def test_mid_page_heading_starts_new_chunk():
    text = f"IRRIGACIÓN\n\n{words(350)}\n\nPROTOCOLO FINAL\n\n{words(40)}"
    chunks = chunk_document(META, doc((text, ["IRRIGACIÓN", "PROTOCOLO FINAL"])), 300, 800)
    assert [c["section_title"] for c in chunks] == ["IRRIGACIÓN", "PROTOCOLO FINAL"]


def test_pdf_lines_without_blank_lines_form_paragraphs():
    text = (
        "IRRIGACIÓN\n"
        "El hipoclorito de sodio es el irrigante principal durante la preparación químico-\n"
        "mecánica. Disuelve tejido orgánico.\n"
        "Las concentraciones varían entre 0,5% y 5,25%.\n"
        "Paso 1: irrigar con hipoclorito\n"
        "durante la instrumentación.\n"
    )
    assert split_blocks(text) == [
        Block("IRRIGACIÓN", is_heading=True),
        Block(
            "El hipoclorito de sodio es el irrigante principal durante la preparación "
            "químico-mecánica. Disuelve tejido orgánico.",
            is_heading=False,
        ),
        Block("Las concentraciones varían entre 0,5% y 5,25%.", is_heading=False),
        Block("Paso 1: irrigar con hipoclorito durante la instrumentación.", is_heading=False),
    ]


def test_list_items_start_new_paragraphs():
    blocks = split_blocks("Indicaciones\n- pulpa necrótica\n- lesión periapical\n")
    assert [b.text for b in blocks] == ["Indicaciones", "- pulpa necrótica", "- lesión periapical"]


def test_docx_lines_are_paragraphs():
    blocks = split_blocks("Primera línea sin punto\nSegunda línea", line_is_paragraph=True)
    assert [b.text for b in blocks] == ["Primera línea sin punto", "Segunda línea"]


def test_split_to_limits_prefers_sentence_boundaries():
    text = " ".join(f"Oración número {i} con cinco." for i in range(10))  # 10 x 5 tokens
    pieces = split_to_limits(text, 12, 12)
    assert all(estimate_tokens(p) <= 12 for p in pieces)
    assert all(p.endswith(".") for p in pieces)
    assert " ".join(pieces) == text


def test_split_to_limits_cuts_overlong_sentence_by_words():
    pieces = split_to_limits(words(25), 10, 10)
    assert [estimate_tokens(p) for p in pieces] == [10, 10, 5]


def test_section_carries_over_to_next_page():
    chunks = chunk_document(
        META, doc(("IRRIGACIÓN\n" + words(780) + ".", []), (words(100) + ".", [])), 300, 800
    )
    assert [(c["page_start"], c["page_end"], c["section_title"]) for c in chunks] == [
        (1, 1, "IRRIGACIÓN"),
        (2, 2, "IRRIGACIÓN"),
    ]


def test_page_gap_ends_the_chunk():
    chunks = chunk_document(
        META, doc((words(50) + ".", []), ("", []), (words(50) + ".", [])), 300, 800
    )
    assert [(c["page_start"], c["page_end"]) for c in chunks] == [(1, 1), (3, 3)]


def test_heading_at_page_end_joins_next_page_content():
    chunks = chunk_document(
        META, doc((words(50) + ".\nPROTOCOLO FINAL", []), ("Paso 1: irrigar.", [])), 300, 800
    )
    assert [(c["page_start"], c["page_end"], c["section_title"]) for c in chunks] == [
        (1, 1, ""),
        (1, 2, "PROTOCOLO FINAL"),
    ]
    assert chunks[1]["text"] == "PROTOCOLO FINAL\n\nPaso 1: irrigar."


def test_heading_plus_long_paragraph_stays_within_max():
    chunks = chunk_document(
        META,
        doc(
            ("OBTURACIÓN\n" + words(900), []),
        ),
        300,
        800,
    )
    assert all(c["token_estimate"] <= 800 for c in chunks)
    assert chunks[0]["text"].startswith("OBTURACIÓN")


def test_heading_only_document_produces_no_chunks():
    assert (
        chunk_document(
            META,
            doc(
                ("INTRODUCCIÓN", []),
            ),
            300,
            800,
        )
        == []
    )
