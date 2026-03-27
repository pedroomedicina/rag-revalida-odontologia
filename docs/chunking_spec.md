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

## Core rules
- Prefer semantic units (paragraph groups), not raw fixed-size slicing.
- Preserve page boundaries (do not merge across non-contiguous pages).
- Start a new chunk on explicit section switch when available.
- Target size:
  - min: ~300 token estimate
  - max: ~800 token estimate
- If a paragraph would overflow max size and current chunk already satisfies min size, flush and start a new chunk.

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
