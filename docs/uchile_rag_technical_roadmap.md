# Technical roadmap — Universidad de Chile grounded study assistant

## Goal
Build a Spanish-only chatbot for exam/revalidation prep that answers **only** from an approved Universidad de Chile corpus, cites exact pages, surfaces conflicts between sources, and refuses unsupported answers.

---

## Current implementation snapshot (this repository)

Pipeline code lives in `src/rag/`; `scripts/` holds thin CLI wrappers. Known defects are tracked in `docs/status_2026-11-19.md`.

### Already implemented
- corpus cleanup with dedupe/filtering/normalization: `src/rag/corpus_cleanup.py`
- inventory generator: `src/rag/generate_inventory_csv.py`
- metadata template and constraints: `configs/metadata_template.json`
- inventory validator with schema/value checks: `src/rag/validate_inventory.py`
- split OCR quality checks (moment 1 + moment 2): `src/rag/ocr_quality_checks.py`
- ingestion with fallback chain (`pypdf` -> repair -> OCR); text-less PDFs are routed to OCR and failed documents go to review: `src/rag/ingest.py`
- stable `doc_id`s derived from the path within the corpus
- chunking with paragraph detection, a hard max size, page-spanning procedures, and mid-page headings: `src/rag/chunker.py` (spec: `docs/chunking_spec.md`)
- Postgres hybrid index (pgvector embeddings + accent-insensitive full-text search): `src/rag/build_retrieval_index.py`, `src/rag/text_search.py`
- hybrid retrieval with reciprocal rank fusion and cross-encoder reranking: `src/rag/retrieve_hybrid.py`
- grounded answer generation with citation checks: `src/rag/answer_with_rag.py`
- orchestration script for prep order: `src/rag/run_mvp_prep.py`
- uv packaging (Python 3.12), pytest suite on a synthetic corpus, Postgres integration tests, and CI

### Pipeline outputs (generated, not tracked)
- `data/processed/corpus_inventory.csv`
- `data/normalized/*.json`
- `data/processed/chunks.jsonl`
- `artifacts/reports/*.json`

### Current state
- prep pipeline, chunking, indexing, retrieval, and answering are implemented end to end
- open defects: index never deletes stale rows on re-chunk, and citation validator gaps
- next active focus: fix the open defects, then build a gold evaluation set and a retrieval recall baseline

---

## 0. Product constraints to lock before coding

These are not code tasks, but they affect every technical choice:

- Answer only from approved corpus
- No web browsing in answer generation
- Spanish only
- Page-level citations required
- Show supporting excerpts
- Handle conflicting criteria explicitly
- Distinguish source authority levels
- Educational prep only, not live professional guidance

**Output of this phase**
- Source authority policy
- Answer contract
- Conflict policy
- Inclusion/exclusion policy for documents

---

## 1. Corpus foundation

### Objective
Prepare the document base so retrieval is reliable.

### Tasks
- Collect an initial MVP corpus
- Prioritize official guides, regulations, syllabi, and official slides
- Exclude unclear-rights material for now
- Create a corpus inventory with metadata:
  - document ID
  - title
  - course/program
  - faculty/career
  - teacher
  - year/semester
  - source type
  - authority level
  - page count
  - OCR quality
  - rights status

### Technical decisions
- Use a canonical file naming convention
- Store original files separately from processed text
- Version every file
- Mark duplicates and replacements

**Deliverables**
- `/data/raw/`
- `/data/processed/`
- `corpus_inventory.csv`
- `source_policy.md`

---

## 2. Ingestion pipeline

### Objective
Turn PDFs/slides/scans into clean, searchable, page-aware text.

### Tasks
- Parse text PDFs directly
- If parsing fails, attempt PDF repair before retrying extraction
- If repair still fails and OCR tooling is available, run OCR fallback
- Run OCR for scanned PDFs
- Preserve page numbers during extraction
- Detect headings/sections where possible
- Extract basic structure:
  - title
  - headings
  - bullet lists
  - tables where feasible
- Save normalized text as JSON or Markdown per page/section

### OCR quality checkpoints (split by moment)
- **Moment 1 (inventory-time heuristic):**
  - run immediately after inventory generation
  - quick signal only (`unknown`/`needs review` candidates)
  - does not block corpus prep by itself
- **Moment 2 (post-ingestion quality):**
  - run after normalized text artifacts exist
  - updates final `ocr_quality` used by ranking and readiness
  - should gate chunking/retrieval when quality is `poor`/`failed`/`unknown`

Recommended gate:
- do not move documents to chunking when moment-2 quality is below threshold

### Output format suggestion
```json
{
  "doc_id": "UCH-ODO-2024-001",
  "title": "Cerámicas y Cementación III",
  "pages": [
    {
      "page": 1,
      "text": "...",
      "sections": ["Introducción", "Clasificación"]
    }
  ]
}
```

### Tools / stack options
- Python
- PyMuPDF / pdfplumber for text PDFs
- OCR pipeline for scanned pages
- Optional document AI later if scans are poor

**Deliverables**
- `ingest.py`
- `ocr_pipeline.py`
- `normalized_docs/*.json`
- ingestion reports with per-document extraction method trace
- inventory updates with final OCR quality after moment 2

---

## 3. Metadata and topic tagging

### Objective
Give retrieval and ranking the context needed to prefer the right source.

### Required metadata
At document level:
- `doc_id`
- `title`
- `university`
- `faculty`
- `career`
- `course`
- `teacher`
- `source_type`
- `authority_level`
- `year`
- `semester`
- `version_date`
- `rights_status`

At chunk level:
- `chunk_id`
- `doc_id`
- `page_start`
- `page_end`
- `section_title`
- `topic_tags`
- `content_type` (`conceptual`, `procedural`, `comparative`, `product-specific`)
- `ocr_confidence`

### Topic taxonomy
Start small and controlled. Example:
- materials
- ceramics
- zirconia
- cementation
- acid-etchable ceramics
- non-acid-etchable ceramics
- primers
- cements

**Deliverables**
- `metadata_schema.md`
- `topic_taxonomy.yaml`
- `tagging_rules.md`

---

## 4. Chunking strategy

### Objective
Split documents in a way that preserves meaning and supports accurate citations.

### Recommendation
Do **not** chunk only by token count.
Use a hybrid strategy:
1. split by heading/subheading
2. preserve page boundaries
3. apply token-based fallback only when sections are too long

### Chunking rules
- Target chunk size: ~300–800 tokens
- Overlap: small overlap only when needed
- Keep procedure steps together when possible
- Never merge unrelated headings into one chunk
- Attach page range to each chunk

### Why this matters
Your users will ask section-level questions, not just keyword queries. Procedural content especially performs much better when chunked by semantic unit.

**Deliverables**
- `chunker.py`
- `chunking_spec.md`
- `chunks.parquet` or `chunks.jsonl`

---

## 5. Retrieval system (RAG core)

### Objective
Find the best supporting passages before generating an answer.

### Recommended architecture (MVP, no wheel reinvention)
Use **hybrid retrieval** with common, battle-tested components:
- embeddings: `sentence-transformers` multilingual model
- vector index/storage: Postgres + `pgvector` (Supabase-compatible)
- keyword index: Postgres full-text search (`tsvector` + GIN)
- rerank: off-the-shelf cross-encoder reranker

This gives you one datastore for metadata + vectors + keyword index, and avoids custom search infra.

### Concrete build steps

#### Step 5.1 — Store chunks in Postgres (Supabase/pgvector)
Create a `rag_chunks` table with:
- identifiers: `chunk_id`, `doc_id`
- provenance: `title`, `filename`, `page_start`, `page_end`, `section_title`
- quality/ranking metadata: `ocr_quality`, `content_type`
- retrieval fields: `text`, `embedding`, `search_tsv`

Indexes:
- `ivfflat` (or `hnsw`) on `embedding`
- `GIN` on `search_tsv`

#### Step 5.2 — Embedding ingestion job
Input: `chunks.jsonl`

Job behavior:
1. load chunk rows
2. embed chunk text in batches
3. upsert into `rag_chunks`
4. verify row counts + sample retrieval sanity checks

Notes:
- start with one stable embedding model for consistency
- version your embedding model in config (`EMBEDDING_MODEL_NAME`)
- re-embed only if model changes

#### Step 5.3 — Keyword retrieval (BM25-ish) using Postgres FTS
For MVP, use Postgres FTS instead of a separate BM25 engine:
- query via `plainto_tsquery('spanish', query)`
- score with `ts_rank_cd(search_tsv, tsquery)`
- retrieve top `k_keyword`

This is usually enough before introducing OpenSearch/Elasticsearch.

#### Step 5.4 — Hybrid retrieve + rerank
Per query:
1. normalize query text (Spanish lowercase/trim)
2. run vector retrieval top `k_vector`
3. run keyword retrieval top `k_keyword`
4. merge candidates with weighted RRF or weighted score sum
5. rerank top `k_merge` using cross-encoder
6. keep top `k_final` as evidence for generation

Recommended defaults for MVP:
- `k_vector=30`
- `k_keyword=30`
- `k_merge=40`
- `k_final=8`

#### Step 5.5 — Pass evidence to generator with citations
Pass only top chunks (never full corpus), with structured provenance:
- `chunk_id`
- `doc_id`
- `title`
- `page_start`, `page_end`
- `text` excerpt
- final score

Generation prompt must require citations in every claim:
- citation format: `[{doc_id} pp.{page_start}-{page_end}]`
- if evidence is insufficient, abstain explicitly

### Retrieval pipeline (end-to-end)
1. user question arrives
2. normalize Spanish query
3. semantic retrieval (`pgvector`)
4. keyword retrieval (FTS)
5. candidate merge (hybrid fusion)
6. rerank (cross-encoder + metadata priors)
7. send top evidence chunks to answer generator
8. return answer + citations + supporting excerpts

### Ranking factors (post-rerank adjustments)
- semantic similarity
- keyword relevance
- authority level
- recency/version
- section-title/query overlap
- OCR quality penalties
- optional course/program filter

### Suggested implementation files (MVP)
- `scripts/build_retrieval_index.py` (embed + upsert + indexes)
- `scripts/retrieve_hybrid.py` (hybrid retrieve + rerank + citation-ready output)
- `scripts/retrieval_smoke_test.py` (sanity checks on known queries)
- `ranking_rules.md`

**Deliverables**
- working hybrid retrieval CLI
- retrieval-ready Postgres schema and indexes
- citation-ready evidence payload for generation layer

---

## 6. Generation layer

### Objective
Produce concise answers grounded only in retrieved evidence.

### Prompt behavior
The model should be instructed to:
- answer only from provided passages
- cite sources by document + pages
- mention when evidence is incomplete
- surface conflicts explicitly
- avoid guessing
- avoid external/general knowledge

### Output contract
Recommended format:

```md
## Respuesta
...

## Fuentes
- Documento A, pp. 4–5
- Documento B, p. 8

## Fragmentos de apoyo
> ...

## Observaciones
- Hay conflicto entre fuentes
- No encontré respaldo suficiente en el corpus
```

### Important rule
The model should never see the entire corpus. Only retrieved chunks + metadata.

**Deliverables**
- `answer_generator.py`
- `prompts/answering.md`
- `response_schema.json`

---

## 7. Conflict detection and authority logic

### Objective
Handle contradictory documents without hiding the disagreement.

### Minimum rules
- If two high-scoring chunks disagree, flag possible conflict
- Prefer higher-authority documents when summarizing
- Do not erase minority/older criteria if still relevant
- Explain which source is newer or more authoritative

### Example behavior
- “La guía docente 2024 indica X.”
- “Una diapositiva 2022 menciona Y.”
- “Dado el nivel de autoridad y fecha, se prioriza X para fines de estudio.”

### Technical implementation
- Add source weights to reranking
- Add contradiction heuristics for retrieved chunks
- Optionally use a second LLM call to compare top evidence

**Deliverables**
- `conflict_policy.md`
- `evidence_compare.py`

---

## 8. Citation system

### Objective
Make every answer auditable.

### Requirements
- Page-level citations mandatory
- Link each citation to exact chunk/page
- Show short support excerpts
- Preserve source titles and metadata

### UX recommendation
Allow the user to expand:
- source page
- excerpt used
- full document metadata

### Backend requirement
Store exact provenance per answer:
- query
- retrieved chunks
- chunk scores
- final cited chunks
- answer text

**Deliverables**
- provenance logging
- citation formatter
- source preview UI/API

---

## 9. Evaluation framework

### Objective
Measure trustworthiness before launch.

### You need a test set
Even if you do not have real student questions yet, start with synthetic ones from the corpus.
Later replace or expand with real user questions.

### Metrics
- retrieval precision
- citation correctness
- answer faithfulness
- conflict handling quality
- abstention quality
- latency

### Evaluation categories
- direct factual lookup
- procedural questions
- comparative questions
- synthesis across 2–3 sources
- conflict cases
- unsupported questions

### Minimum test set for MVP
- 50–100 questions
- 10+ conflict/ambiguity cases
- 10+ unsupported cases

**Deliverables**
- `eval_questions.csv`
- `gold_answers.md`
- automated eval scripts

---

## 10. Admin tooling

### Objective
Make the corpus maintainable.

### Features
- upload document
- assign metadata
- mark authority level
- preview OCR quality
- reprocess document
- approve/reject document for production
- view duplicates/version history

### Nice to have
- document diffing between versions
- topic tag editor
- conflict review queue

**Deliverables**
- admin panel MVP
- ingestion status dashboard

---

## 11. User-facing application

### MVP features
- Spanish chat UI
- answer with citations
- source panel / excerpt panel
- “not found in corpus” state
- feedback buttons: useful / not useful / wrong citation
- optional document/course filters

### Good MVP UX choices
- keep answers short first, expandable later
- show confidence carefully; do not overclaim
- let users inspect sources easily

### Avoid in MVP
- user file uploads
- memory across users
- free-form web search
- voice mode
- multi-language support

**Deliverables**
- web app
- auth/basic billing if needed
- analytics events

---

## 12. Logging, feedback, and improvement loop

### Objective
Use real usage to improve retrieval and coverage.

### Log
- question
- answer
- sources used
- thumbs up/down
- manual correction status
- unanswered queries
- repeated failed queries

### Use feedback to
- add missing docs
- retag chunks
- improve retrieval weights
- refine prompt behavior
- identify high-value topics for academies

**Deliverables**
- analytics pipeline
- review queue
- feedback dashboard

---

## 13. Security and data handling

### Since this is academic content only
The risk is lower than patient-facing systems, but you still need good hygiene.

### Minimum controls
- private document storage
- role-based admin access
- signed URLs or protected file access
- encrypted secrets
- audit trail for corpus updates
- clear data retention rules for chat logs

### Keep MVP simple
- no user uploads
- no unnecessary personal data
- basic account and subscription data only

**Deliverables**
- security checklist
- backup policy
- access control config

---

## 14. Suggested build order

### Phase 0 — Corpus ops foundation (already done in this repo)
**Goal:** establish deterministic corpus preparation and metadata control

Build:
- cleanup + dedupe workflow
- inventory generation
- metadata template
- validation gates
- split OCR checks and orchestration

**Exit criteria:**
- inventory generated reproducibly
- validator passes on required metadata structure
- moment-1 OCR report generated

### Current checkpoint (Nov 2026)
- done: cleanup + inventory + validation + OCR moment 1 and moment 2
- done: ingestion with OCR routing for text-less PDFs, stable `doc_id`s
- done: chunking (`chunks.jsonl`), hybrid index, hybrid retrieval + rerank, answer generation with citations
- done: test suite, Postgres integration tests, CI
- next: delete-by-doc before upsert, citation validator fixes, gold evaluation set, recall baseline

### Phase 1 — Proof of concept
**Goal:** prove answer quality on a small corpus

Build:
- chunking
- hybrid retrieval
- basic answer generation with citations
- manual evaluation set

**Exit criteria:**
- answers grounded in sources
- page citations correct most of the time
- unsupported questions refused cleanly

### Phase 2 — MVP
**Goal:** usable product for early testers

Build:
- admin upload workflow
- metadata tooling
- conflict handling
- user chat UI
- provenance logging
- feedback collection

**Exit criteria:**
- 50–100 evaluated questions
- stable response format
- clear source previews
- acceptable latency

### Phase 3 — Pilot with students / academy
**Goal:** validate demand and trust

Build:
- billing or seat management
- analytics dashboard
- usage monitoring
- improved ranking rules
- partial course filters

**Exit criteria:**
- real user retention
- repeated use before exams
- manageable support burden

### Phase 4 — Scale and hardening
**Goal:** support more corpus breadth and better operations

Build:
- larger corpus ingestion
- better OCR fallback
- version diffing
- more robust eval harness
- automated regression testing
- more advanced admin workflows

---

## 15. Practical MVP stack suggestion

### Backend
- Python + FastAPI
- Postgres
- pgvector
- BM25 / full-text search layer
- object storage for documents

### Processing
- PyMuPDF / pdfplumber
- OCR service or local OCR pipeline
- background workers for ingestion jobs

### Frontend
- Next.js or similar
- source sidebar with citations/excerpts

### Model usage
- one model call for answering
- optional smaller/cheaper helper calls for tagging or conflict comparison

### Why this stack
It is simple, common, and enough for an MVP before over-optimizing.

---

## 16. Team / role needs

Minimum realistic setup:
- 1 technical builder
- 1 domain reviewer or academic collaborator
- 1 person for corpus ops/metadata/rights tracking

If solo:
- narrow the corpus aggressively
- avoid fancy UI first
- invest early in evals, not just demo quality

---

## 17. 8-week MVP example timeline

### Week 1
- define policies
- prepare corpus with cleanup + inventory + validation
- run OCR check moment 1
- freeze first MVP corpus slice

### Week 2
- build ingestion + OCR pipeline
- normalize documents into page-aware artifacts
- run OCR check moment 2 and update inventory

### Week 3
- implement chunking + metadata tagging
- store chunks
- validate chunk citation mapping to source pages

### Week 4
- build hybrid retrieval
- test search quality manually

### Week 5
- implement answer generation + citation formatting
- add abstention behavior

### Week 6
- build basic UI + source panel
- create first evaluation set

### Week 7
- add conflict handling + logging + feedback
- improve ranking rules

### Week 8
- run pilot with a few testers
- identify failure modes
- decide whether to expand corpus or refine retrieval first

---

## 18. Biggest technical failure modes to expect

- bad OCR on scans
- missing or inconsistent metadata
- poor chunking that separates procedure steps incorrectly
- citations that point to the wrong page
- model over-summarizing weak evidence
- hidden conflicts between sources
- duplicate/outdated documents polluting retrieval

Plan for these early. They are more important than model sophistication.

---

## 19. Recommendation for your exact case

Start with a **single career or subject area**, not the whole institution.

Best path:
1. keep the current narrow corpus and inventory pipeline as baseline
2. implement ingestion with stable `doc_id` mapping to normalized outputs
3. run OCR quality moment 2 and mark low-quality docs before chunking
4. implement chunking with page-range preservation
5. generate embeddings + hybrid retrieval
6. evaluate with 50 questions
7. only then expand breadth

That will teach you more than trying to ingest hundreds of guides immediately.

---

## 20. Definition of success for MVP

The MVP is successful if a student can ask a question in Spanish and the system:
- retrieves the relevant Universidad de Chile material,
- answers briefly and clearly,
- cites exact pages,
- shows support excerpts,
- surfaces conflicts when needed,
- and refuses unsupported questions in