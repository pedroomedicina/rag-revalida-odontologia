# Ranking rules (Section 5 MVP)

## Goal
Select the most relevant and trustworthy chunks before generation, while preserving citation fidelity.

## Candidate generation
- vector retrieval from `pgvector` (`k_vector`)
- keyword retrieval from Postgres FTS (`k_keyword`)
- merge using weighted Reciprocal Rank Fusion (RRF)

## Fusion formula
- `hybrid_score = vector_weight * (1 / (rrf_k + vector_rank)) + keyword_weight * (1 / (rrf_k + keyword_rank))`
- defaults:
  - `vector_weight = 0.55`
  - `keyword_weight = 0.45`
  - `rrf_k = 60`

## Reranking
- apply cross-encoder reranker to merged candidates (`k_merge`)
- default model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

## Final score
- `final_score = rerank_score + 0.2 * hybrid_score + quality_prior`

## Quality prior by OCR quality
- `good`: `+0.05`
- `usable_with_caution`: `-0.02`
- `poor`/`failed`/`unknown`: `-0.10`

## Top-k defaults
- `k_vector=30`
- `k_keyword=30`
- `k_merge=40`
- `k_final=8`

## Citation rules
- each returned chunk must include citation:
  - single page: `[DOC_ID p.X]`
  - range: `[DOC_ID pp.X-Y]`
- generation layer must cite only returned chunks

## Tuning process
- evaluate on a fixed query set
- tune one parameter at a time
- prioritize citation correctness and retrieval recall over minor latency gains
