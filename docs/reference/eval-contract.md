# Eval contract

Names, formats, and pass rules used by [../how-to/sessions.md](../how-to/sessions.md). Change them by editing this file before changing the scripts, so the reports stay comparable.

## Question file

Path: `evals/questions.jsonl`. One JSON object per line.

```json
{"id":"endo-001","bucket":"lookup","question":"¿Cuándo se indica el hidróxido de calcio en endodoncia?","gold":[{"doc_id":"UCH-ENDO-012","page_start":4,"page_end":5}],"must_answer":true,"gold_fact":"sentence copied from that page"}
```

| Field | Rule |
| --- | --- |
| `id` | Stable. Do not renumber after the first report. |
| `bucket` | `lookup`, `procedure`, `paraphrase`, or `unsupported` |
| `gold` | Page spans. Empty array when `must_answer` is false. |
| `must_answer` | True only when you opened a page that contains `gold_fact`. |
| `gold_fact` | A sentence copied from that page. Empty when the question must be refused. |

First report: 20 questions, five per bucket. Résumé report: 40, ten per bucket. Add the second 20 only after `docs/failure.md` exists.

Gold pages come from the normalized documents or from a full-text search you then open. They do not come from the current top 8. See [../decisions/0004-label-gold-from-source-pages.md](../decisions/0004-label-gold-from-source-pages.md).

## Retrieval hit

Used for every `must_answer: true` question. Unsupported questions are not part of recall.

A chunk hits a gold span when `doc_id` matches and the page ranges overlap:

```python
chunk["page_start"] <= gold["page_end"] and gold["page_start"] <= chunk["page_end"]
```

A question is a hit when any returned chunk hits any of its gold spans. Recall at 8 is hits divided by answerable questions, with `--k-final 8`.

Write the result as a count and a percentage: `14/15`. A bucket of five makes one miss worth 20 points. The count is what you say out loud.

Modes in every retrieval report: `vector`, `keyword`, `hybrid`, `hybrid_rerank`.

## Retrieval report

| File | When |
| --- | --- |
| `artifacts/reports/eval_retrieval_baseline.json` | Session 3, before any retrieval edit |
| `artifacts/reports/eval_retrieval_probes.json` | Session 4, probes only |
| `artifacts/reports/eval_retrieval_prefixes.json` | Session 4, after reindex, probes frozen |

Each file records: git commit, chunk count, embedding model, reranker model, whether prefixes were on, `k_vector`, `k_keyword`, `k_merge`, `k_final`, fusion weights, RRF constant, `ivfflat.probes`, hits per mode, hits per mode per bucket.

Copy the baseline aside and commit it before Session 4 starts.

## Answer pass

Run answers on `hybrid_rerank` only, temperature `0`, prompt file `prompts/answer_v1.txt`.

| Question | Pass |
| --- | --- |
| `must_answer: true` | `suficiencia_evidencia` is `suficiente`, citation validation returns no errors, and a cited chunk overlaps the gold page |
| `must_answer: false` | `suficiencia_evidencia` is `insuficiente` and `fuentes` is empty |

Citation validation is the check already in `answer_with_rag.py`: every `evidence_id` is one you retrieved, and the citation string matches that chunk.

## Run log

`artifacts/reports/runs.jsonl`, one line per answered question:

- question id
- mode
- prompt sha256
- latency for embed, database, rerank, and generation
- model name
- token count
- sufficiency
- citations
- pass or fail

## Threshold

`evals/thresholds.json`:

```json
{"hybrid_rerank_min_hits": 14, "answerable": 15}
```

Replace the numbers with the baseline you intend to keep. `eval_retrieval.py` exits non-zero when a later run is below `hybrid_rerank_min_hits`. Update the file only when you accept a new baseline on purpose. Session 7 proves the gate by setting `k_final` to 1 and watching the command fail.

## Failure note

`docs/failure.md` in the repo, after one change:

- question id and text
- what the system returned
- the cause (chunk boundary, missing from top 8, or a citation on the wrong retrieved page)
- the change
- the retrieval count before and after, or the answer count before and after, matching what you changed
