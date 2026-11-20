# Sessions

Do these in order. Each session names the files it touches and the condition that ends it. Each session is a GitHub issue in the *Proof of concept* or *MVP page* milestone. Close it only when that condition is true.

Course PDFs stay out of git. Pipeline code lives in `src/rag/`; each module has a thin CLI wrapper in `scripts/`.

Contracts: [../reference/eval-contract.md](../reference/eval-contract.md). Diagram: [../reference/request-path.md](../reference/request-path.md).

## Session 0. One real question

Run `uv sync --extra rag` and set `DATABASE_URL` to Postgres with pgvector.

If `data/processed/chunks.jsonl` is missing, run:

```bash
uv run python scripts/run_mvp_prep.py /path/to/corpus
uv run python scripts/build_retrieval_index.py --dsn "$DATABASE_URL"
```

Ask one endodontics question through retrieval and through answering:

```bash
uv run python scripts/retrieve_hybrid.py --query "Tu pregunta"
uv run python scripts/answer_with_rag.py --query "Tu pregunta" --strict-citations
```

Open the cited page. Write `docs/baseline.md` with the chunk count, the question, the top citation, and whether that page supports the answer.

```sql
SELECT count(*) FROM rag_chunks;
```

Done when `docs/baseline.md` exists, including a wrong citation if that is what you found.

## Session 1. Twenty gold questions

Search the documents, then open the page, then write the row. Do this before you trust retrieval.

```bash
rg -n "hidróxido de calcio" data/normalized
```

```sql
SELECT doc_id, page_start, page_end, left(text, 180)
FROM rag_chunks
WHERE search_tsv @@ plainto_tsquery('spanish', 'hidróxido de calcio')
LIMIT 20;
```

The SQL hit is a candidate. The gold span is the page you opened. Append to `evals/questions.jsonl` using the schema in the eval contract.

Five questions in each bucket:

- `lookup`: the page's own terms.
- `procedure`: a step sequence. The gold range covers the steps, not only the heading.
- `paraphrase`: different words from the heading. The page still contains the fact.
- `unsupported`: `gold` is `[]` and `must_answer` is false. Keep the row only after both searches return nothing.

Done when the file has 20 lines and every answerable row names a page you opened.

## Session 2. The mode flag

Draw the request path once, on paper, before editing.

In `src/rag/retrieve_hybrid.py`, move the body of `main` into `retrieve(...)` and return the same JSON dict. Add `--mode` with default `hybrid_rerank`.

- `vector`: `fetch_vector_hits`, sort by `vector_score`, cut to `--k-final`.
- `keyword`: `fetch_keyword_hits`, cut to `--k-final`.
- `hybrid`: `hybrid_fuse`, cut to `--k-final`.
- `hybrid_rerank`: fuse, cut to `--k-merge`, rerank, apply `apply_quality_prior`, cut to `--k-final`.

Construct `SentenceTransformer` only when the mode queries vectors. Construct `CrossEncoder` only for `hybrid_rerank`. Put `mode` on the `retrieval` object. Format citations in every mode.

Done when one question has a different top citation for `vector` and for `keyword`, and `hybrid_rerank` still cites a real page.

## Session 3. The retrieval score

Add `src/rag/eval_retrieval.py` with a `scripts/` wrapper. It imports `retrieve`, builds each model once, and passes the instances in. It does not start a subprocess per question.

For every answerable question and each mode, apply the page-overlap hit rule from the eval contract. Leave unsupported questions out of the score.

Write `artifacts/reports/eval_retrieval_baseline.json` with the fields listed in the contract. Print counts as `hybrid_rerank 14/15`, overall and per bucket.

Done when that file exists and you can see which mode won `lookup` and which won `paraphrase`.

## Session 4. One change, then a second

Commit the baseline report before editing retrieval.

If vector hits are far below keyword hits, add an integer `--probes` flag. Before the vector query, set `ivfflat.probes` to that integer. Re-run. Save `artifacts/reports/eval_retrieval_probes.json`.

After that file is saved, prefix embeddings. Index with `passage: ` in front of the chunk text. Query with `query: ` in front of the normalized question. Reindex. Save `artifacts/reports/eval_retrieval_prefixes.json` at the probe value you just chose.

Leave the fusion weights alone.

If vector recall on the baseline was already close to hybrid, skip this session and write that sentence at the top of the baseline report's notes.

Done when the two new report files exist, or the skip sentence is written.

## Session 5. Grade answers on one mode

Move the system prompt into `prompts/answer_v1.txt`. Record its sha256 on each run.

Add `src/rag/eval_answers.py` with a `scripts/` wrapper. Run it only on `hybrid_rerank`, temperature `0`. Apply the pass rules in the eval contract. Append `artifacts/reports/runs.jsonl`.

Done when you have a pass count, such as `11/20`, and the misses listed by question id.

## Session 6. Fix one miss

Pick the miss whose cause you can name.

- A procedure split across chunks.
- A gold page absent from the top 8.
- A citation that points at a retrieved chunk on the wrong page.

Change that one thing. Re-run Session 3 if you touched retrieval or chunking. Re-run Session 5 if you touched the prompt or the citation check. Write `docs/failure.md` with the question, what came back, the cause, the change, and both scores.

Done when `docs/failure.md` quotes the two counts.

## Session 7. Lock the score

Add `evals/thresholds.json` with the hit count you will keep. `eval_retrieval.py` exits non-zero below that count.

Prove it: run with `--k-final 1` and confirm the command fails. Restore `--k-final 8`.

Done when that failure happens and the real run still passes.

## Session 8. A page

Start this only after `docs/failure.md` exists.

- FastAPI `POST /answers` returns the current JSON plus the trace.
- One Spanish page shows the answer, the citations, the excerpts, and the refusal.
- "Esto está mal" appends the question and answer id to `feedback.jsonl`.
- `docker-compose` starts Postgres with pgvector.
- Three short pages you wrote live in the repo so the eval runs without the course PDFs.
- The README opens with the request-path diagram, the hit table, and the failure note.

Done when someone else can boot the sample, ask a question, and see a citation or a refusal.

## After Session 6

Add five more questions per bucket, labeled the same way, until the file has 40. Re-run retrieval and answers. The résumé number is that run. Update `evals/thresholds.json` only if you are accepting the new count as the baseline.
