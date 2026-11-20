# rag-revalida-odontologia

MVP RAG pipeline for Spanish endodontics study material. The project prepares a corpus, validates metadata, creates chunked artifacts, and supports hybrid retrieval plus answer generation.

Plan, design, and decisions: [docs/README.md](docs/README.md). Work items: GitHub Issues and the project board. Dated audit: [docs/status_2026-11-19.md](docs/status_2026-11-19.md).

## Project layout

- `src/rag/`: pipeline logic (cleanup, inventory, OCR checks, ingestion, chunking, indexing, retrieval, answering).
- `scripts/`: thin CLI wrappers, one per `src/rag` module. Each is also runnable as `python -m rag.<module>`.
- `tests/`: pytest suite. `tests/fixtures/corpus_builder.py` generates a small synthetic Spanish corpus, so the pipeline is tested without course PDFs.
- `configs/`: metadata and validation templates.
- `data/raw/`: source corpus (not tracked).
- `data/normalized/`: normalized JSON artifacts from ingestion (not tracked).
- `data/processed/`: generated inventory and chunk outputs (not tracked).
- `artifacts/reports/`: pipeline reports (not tracked).
- `artifacts/indexes/`: optional local index artifacts (not tracked).
- `docs/`: project documentation.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if it is missing.

Prep pipeline and tests only (light, no torch):

```bash
uv sync
```

With retrieval, indexing, and answering (adds torch, sentence-transformers, psycopg, openai):

```bash
uv sync --extra rag
```

Configure environment:

```bash
cp .env.example .env
```

## Run the prep pipeline

```bash
uv run python scripts/run_mvp_prep.py /path/to/corpus
```

The default outputs are written to:

- `data/processed/corpus_inventory.csv`
- `data/normalized/`
- `data/processed/chunks.jsonl`
- `artifacts/reports/*.json`

## Retrieval and answering

Needs `uv sync --extra rag` and `DATABASE_URL` pointing at Postgres with pgvector.

- Build/update Postgres hybrid index:

```bash
uv run python scripts/build_retrieval_index.py --dsn "$DATABASE_URL"
```

- Run retrieval:

```bash
uv run python scripts/retrieve_hybrid.py --query "Tu pregunta" --dsn "$DATABASE_URL"
```

- Run full RAG answering:

```bash
uv run python scripts/answer_with_rag.py --query "Tu pregunta"
```

## Development

```bash
uv run pytest
```

```bash
uv run ruff check src scripts tests
```

```bash
uv run ruff format src scripts tests
```

CI (`.github/workflows/ci.yml`) runs lint, format check, and tests on every push to `main` and every pull request.

Tests marked `known_bug` describe correct behavior for a defect listed in the status doc. They are strict expected failures: when a fix lands, the test passes unexpectedly, the suite fails, and the marker must be removed in the same change.

## Notes

- Keep large corpus data and generated artifacts out of git.
- Use `scripts/retrieval_smoke_test.py` for quick checks after retrieval changes. Pass retrieval flags after `--`, for example `-- --dsn "$DATABASE_URL"`.
