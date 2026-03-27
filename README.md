# rag-revalida-odontologia

MVP RAG pipeline for Spanish endodontics study material. The project prepares a corpus, validates metadata, creates chunked artifacts, and supports hybrid retrieval plus answer generation.

## Project layout

- `scripts/`: CLI entrypoints for cleanup, ingestion, chunking, indexing, retrieval, and smoke tests.
- `configs/`: metadata and validation templates.
- `data/raw/`: source corpus (not tracked).
- `data/normalized/`: normalized JSON artifacts from ingestion (not tracked).
- `data/processed/`: generated inventory and chunk outputs (not tracked).
- `artifacts/reports/`: pipeline reports (not tracked).
- `artifacts/indexes/`: optional local index artifacts (not tracked).
- `docs/`: project documentation.

## Quick start

1) Create and activate a virtual environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Configure environment:

```bash
cp .env.example .env
```

4) Run the end-to-end prep pipeline:

```bash
python scripts/run_mvp_prep.py /path/to/corpus
```

The default outputs are written to:

- `data/processed/corpus_inventory.csv`
- `data/normalized/`
- `data/processed/chunks.jsonl`
- `artifacts/reports/*.json`

## Retrieval and answering

- Build/update Postgres hybrid index:

```bash
python scripts/build_retrieval_index.py --dsn "$DATABASE_URL"
```

- Run retrieval:

```bash
python scripts/retrieve_hybrid.py --query "Tu pregunta" --dsn "$DATABASE_URL"
```

- Run full RAG answering:

```bash
python scripts/answer_with_rag.py --query "Tu pregunta"
```

## Notes

- Keep large corpus data and generated artifacts out of git.
- Use `scripts/retrieval_smoke_test.py` for quick quality checks after retrieval changes.
