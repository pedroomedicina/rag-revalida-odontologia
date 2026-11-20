# One change per saved report

Status: accepted
Date: 19 November 2026

## Context

Two known differences can move vector recall: IVFFlat probes, and the missing `query:` / `passage:` prefixes on `intfloat/multilingual-e5-base`. Changing both before saving a report makes the delta unreadable. The baseline index is overwritten by a reindex.

## Decision

Save `eval_retrieval_baseline.json` and commit it before editing retrieval. Then probes, saved as its own report. Then prefixes and a reindex, saved as a third report, at the probe value already chosen. If baseline vector recall is already close to hybrid, skip both and write that down.

## Consequences

Session 4 has two report files or an explicit skip. It does not have a single mixed experiment.
