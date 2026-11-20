# Measure retrieval before building an interface

Status: accepted
Date: 19 November 2026

## Context

The repo smoke test checks that a query returns at least three rows and that those rows contain a citation string. An interview asks whether the right page came back. The existing roadmap also lists an admin panel, billing, accounts, and a student pilot.

## Decision

Recall at 8 on labeled pages, an answer grade, and one written fix come before any interface. The interface is one Spanish page and is Session 8. Admin, billing, accounts, uploads, web search, and a pilot stay out of this pack.

Answer generation is graded on `hybrid_rerank` only, after a retrieval report exists. Retrieval eval loads models once in-process.

## Consequences

Session 8 is blocked on `docs/failure.md`. Fusion weights stay at the values in `docs/ranking_rules.md` until a saved report names a reason to change them.
