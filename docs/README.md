# Docs

Plan, contracts, and decisions for the endodontics study assistant. Work items live in GitHub Issues and the project board; these files hold the reasoning the issues link to.

## How this folder is organized

Two conventions, both ordinary in software teams.

**[Diátaxis](https://diataxis.fr/)** splits documents by the job they do:

| Folder | Job | Read it when |
| --- | --- | --- |
| [`explanation/`](explanation/) | Why this product, what finished means | You need to remember the point |
| [`reference/`](reference/) | Diagram, modes, metric, file formats | You are writing the eval or explaining the system |
| [`how-to/`](how-to/) | Sessions, in order, with a stop condition | You are about to do the work |
| [`decisions/`](decisions/) | Choices the sessions do not reopen | You are tempted to change scope |

**Architecture decision records** (one decision per file, status on the first line) are the constraint list. If a session and a decision disagree, the decision wins until a new record supersedes it.

## Read in this order

1. [explanation/design.md](explanation/design.md)
2. [reference/request-path.md](reference/request-path.md)
3. [reference/eval-contract.md](reference/eval-contract.md)
4. [how-to/sessions.md](how-to/sessions.md)

Start at Session 0. Do not start by drawing a second diagram or by opening a UI task.

## Other documents

| File | What it is |
| --- | --- |
| [chunking_spec.md](chunking_spec.md) | Chunk boundaries, sizes, and schema |
| [ranking_rules.md](ranking_rules.md) | Fusion, rerank, and OCR prior defaults |
| [status_2026-11-19.md](status_2026-11-19.md) | Dated audit of environment, code, and a sample-corpus run. Open items are tracked as issues. |
| [uchile_rag_technical_roadmap.md](uchile_rag_technical_roadmap.md) | Original March 2026 roadmap. Background only; the plan above supersedes its build order. |

## Tracking

- **Issues** are the work items: plan sessions, known bugs (each has a `known_bug` test), infra, and refactors.
- **Milestones**: *Proof of concept* (sessions 0–7 and the bugs that block them) and *MVP page* (session 8 and the infra it needs).
- A PR that says `Fixes #N` closes its issue on merge.
