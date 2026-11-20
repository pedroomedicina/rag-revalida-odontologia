# Keep the endodontics corpus

Status: accepted
Date: 19 November 2026

## Context

The private repo already ingests a Spanish endodontics corpus, checks OCR quality, chunks with page ranges, and runs hybrid retrieval plus cited answers. A new chatbot would restart that work.

## Decision

The interview product is this study assistant. The corpus stays endodontics. Breadth beyond that subject waits until the 40-question eval and `docs/failure.md` exist.

## Consequences

Sessions run inside `rag-revalida-odontologia`. New product ideas are out of scope for this pack.
