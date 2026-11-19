# Chunking spec (MVP)

## Objective
Produce semantically coherent, page-aware chunks from `data/normalized/*.json` for hybrid retrieval and citation-ready answering.

## Input
- `data/processed/corpus_inventory.csv`
- `data/normalized/<doc_id>.json`

## Output
- `data/processed/chunks.jsonl` (one JSON object per chunk)
- `artifacts/reports/chunking_report.json`

## Filters
- only include documents with:
  - `status in ["ready"]`
  - `ocr_quality in ["good", "usable_with_caution"]`

## Block detection
Page text is split into heading blocks and paragraph blocks (`split_blocks` in `src/rag/chunker.py`):
- A heading is a line matching `HEADING_RE` (uppercase line, optionally numbered, at most 120 characters). It is a block of its own wherever it appears on the page.
- pypdf output has no blank lines between paragraphs, so a paragraph ends at:
  - a blank line,
  - a line ending in `. ! ? : ; » " )`,
  - or right before a list item (`-`, `•`, `1.`, `a)`, `Paso N`).
- Wrapped lines are joined with a space. A line ending in `-` followed by a lowercase word is joined without the line break (`químico-` + `mecánica` → `químico-mecánica`).
- DOCX extraction: every line is already a paragraph.

## Core rules
- Prefer semantic units (paragraph groups), not raw fixed-size slicing.
- A heading always starts a new chunk and becomes its `section_title`. The section carries over to later pages until the next heading. A heading at the end of a page joins the content on the next page.
- A chunk continues onto the next page only when pages are contiguous (so a procedure that straddles a page break stays together, cited as `pp.X-Y`). A page gap (e.g. an empty or image-only page) ends the chunk.
- Max size is a hard limit: a block that would overflow the current chunk starts a new one, and a single paragraph longer than max is split at sentence boundaries (by words if one sentence is longer than max).
- Target size:
  - min: ~300 token estimate (soft: chunks are shorter only at a heading, a page gap, the end of the document, or before a paragraph that would overflow max)
  - max: ~800 token estimate (hard)

## Chunk schema
- `chunk_id`
- `doc_id`
- `title`
- `filename`
- `page_start`
- `page_end`
- `section_title`
- `content_type`
- `ocr_quality`
- `extraction_method`
- `token_estimate`
- `text`

## Notes
- Token estimate is approximate (whitespace token count), good enough for MVP chunk sizing.
- Post-chunk QA should sample:
  - procedure continuity
  - section coherence
  - citation page accuracy
