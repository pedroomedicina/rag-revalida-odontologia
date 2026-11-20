# Design: endodontics study assistant

Status: accepted for execution. Date: 19 November 2026.

## Problem

Applied AI engineer interviews ask whether you own the path where a model can be wrong. A library list does not answer that. A measured system does: the right page came back, the answer cited it, an unsupported question was refused, and one miss forced a change you can show.

## Product

A Spanish study assistant over an approved endodontics corpus for Universidad de Chile revalidation prep. The student asks a question. The system answers from retrieved pages, cites those pages, and refuses when the corpus does not support an answer.

The code already lives in `pedroomedicina/rag-revalida-odontologia`. Building a second product would drop the part that is hard to fake: a real Spanish corpus, page-aware ingestion, OCR checks, and a hybrid index.

## What already exists

- Corpus cleanup, inventory, metadata validation, OCR quality checks, ingestion (`src/rag/run_mvp_prep.py` and the scripts it calls).
- Chunking spec and `src/rag/chunker.py`.
- Postgres index: `pgvector` embeddings plus a stored Spanish `tsvector` over title, section, and text (`src/rag/build_retrieval_index.py`).
- Hybrid retrieval: vector top 30, full-text top 30, weighted reciprocal rank fusion, cross-encoder, OCR quality prior, top 8 (`src/rag/retrieve_hybrid.py`).
- Answers as JSON, with abstention when retrieval is empty and an optional citation check (`src/rag/answer_with_rag.py --strict-citations`).
- A smoke test that checks row count and that a citation string exists (`src/rag/retrieval_smoke_test.py`). It does not check that the page is the right one.

Defaults in that code:

- Embedding: `intfloat/multilingual-e5-base`, normalized, no `query:` / `passage:` prefix.
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Fusion: vector weight 0.55, keyword weight 0.45, RRF constant 60.
- Final score: rerank score + 0.2 × hybrid score + OCR prior.
- Vector index: IVFFlat, `lists = 100`, cosine.
- Generator: `gpt-4.1-mini`.
- Full-text config: `spanish_unaccent` (Spanish stemming after accent stripping); the question's words are ORed (`src/rag/text_search.py`).

## What this effort adds

Eight behaviors. The first four are mostly in the repo. The last four are the work.

1. A frozen corpus manifest: document ids, page counts, OCR quality, and a hash. Course PDFs stay private.
2. Page-aware chunks, so a citation names a page you can open.
3. Hybrid retrieval with the four modes in [../reference/request-path.md](../reference/request-path.md).
4. An answer bound to those chunks, or a refusal.
5. A labeled question set. First score at 20 questions, five per bucket. Résumé number at 40.
6. Recall at 8 on page overlap, four modes, saved as a report before any change.
7. A trace per answer: stage latency, model, prompt hash, tokens, citations.
8. One written fix with the score before and after.

## What waits

Admin upload, billing, accounts, user uploads, web search, a student pilot, fusion-weight tuning, and a conflict bucket. Conflict questions start only when two source pages disagree and you can point at both. [../decisions/0002-eval-before-interface.md](../decisions/0002-eval-before-interface.md) records this.

The user-facing surface is one page, and it is Session 8. It shows the answer, the citations, the excerpts, and the refusal.

## Done

A stranger can boot a three-page sample you wrote, ask a question, and see a citation or a refusal. The README opens with the request-path diagram, the hit table, and `docs/failure.md`.

You can defend this sentence with the 40-question files:

> Spanish study assistant over an endodontics corpus. Postgres full-text search plus pgvector, fused and reranked. Recall at 8 on a page-labeled set. Unsupported questions refused. One miss written up with the change that moved the score.

## Interview walkthrough

Three queries, then the failure note.

1. A lookup where full-text hit and vectors missed.
2. A paraphrase where vectors hit and full-text missed.
3. A refusal.

## Where to go next

Diagram and mode meanings: [../reference/request-path.md](../reference/request-path.md). Metric and file formats: [../reference/eval-contract.md](../reference/eval-contract.md). Steps: [../how-to/sessions.md](../how-to/sessions.md).
