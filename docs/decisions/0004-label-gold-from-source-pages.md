# Label gold pages from the source, and keep the corpus private

Status: accepted
Date: 19 November 2026

## Context

Recall is only meaningful if the gold page was chosen without looking at the system’s top 8. The Universidad de Chile materials are course documents and are already gitignored as `data/raw`.

## Decision

Every answerable row in `evals/questions.jsonl` is written after the source page is opened. Full-text search may suggest a page. It does not set the gold span. Unsupported rows are kept only when both the normalized JSON and the full-text index have no supporting page.

Course PDFs stay out of git. The public sample in Session 8 is three short pages written for that purpose. The README states that the real corpus is private course material.

## Consequences

A question set built by copying the retriever’s current hits is invalid and gets rewritten. Published eval questions and metrics are allowed. Published course files are not.
