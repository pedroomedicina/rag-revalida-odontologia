#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_QUERIES = [
    "¿Cuándo se indica el hidróxido de calcio en endodoncia?",
    "¿Qué función tiene el hipoclorito de sodio en la preparación químico mecánica?",
    "¿Cuál es la diferencia entre biopulpectomía y necropulpectomía?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run basic smoke tests for hybrid retrieval output quality."
    )
    parser.add_argument(
        "--retrieve-script",
        type=Path,
        default=Path("scripts/retrieve_hybrid.py"),
        help="Path to retrieve_hybrid.py",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Optional text file (one query per line)",
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=3,
        help="Minimum expected retrieval results per query",
    )
    parser.add_argument(
        "--k-final",
        type=int,
        default=8,
        help="Final result count sent to retrieve script",
    )
    parser.epilog = (
        "Any other flags (e.g. --dsn ... --table ...) are passed through to retrieve_hybrid.py."
    )
    return parser


def load_queries(path: Path | None) -> list[str]:
    if not path:
        return DEFAULT_QUERIES
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def run_query(
    retrieve_script: Path, query: str, min_results: int, k_final: int, extra_args: list[str]
) -> bool:
    cmd = [
        sys.executable,
        str(retrieve_script),
        "--query",
        query,
        "--k-final",
        str(k_final),
        *extra_args,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: query failed: {query}")
        if result.stderr.strip():
            print(result.stderr.strip())
        return False

    try:
        payload = json.loads(result.stdout)
    except Exception:
        print(f"FAIL: non-JSON output for query: {query}")
        print(result.stdout[:400])
        return False

    rows = payload.get("results", [])
    if len(rows) < min_results:
        print(f"FAIL: expected >= {min_results} results, got {len(rows)} for query: {query}")
        return False

    missing_citations = [item for item in rows if not str(item.get("citation", "")).strip()]
    if missing_citations:
        print(f"FAIL: missing citation field for query: {query}")
        return False

    print(f"PASS: {query} (results={len(rows)})")
    return True


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse own flags; every unrecognized flag (e.g. --dsn, --table) goes to retrieve_hybrid.

    Both `--dsn X` and the older `-- --dsn X` form work.
    """
    args, unknown = build_parser().parse_known_args(argv)
    if unknown[:1] == ["--"]:
        unknown = unknown[1:]
    args.extra_args = unknown
    return args


def main() -> int:
    args = parse_cli()
    retrieve_script = args.retrieve_script.expanduser().resolve()
    if not retrieve_script.exists():
        raise SystemExit(f"retrieve script not found: {retrieve_script}")

    queries = load_queries(args.queries_file.expanduser().resolve() if args.queries_file else None)
    if not queries:
        raise SystemExit("No queries to test.")

    ok = 0
    for query in queries:
        if run_query(
            retrieve_script=retrieve_script,
            query=query,
            min_results=max(args.min_results, 1),
            k_final=max(args.k_final, 1),
            extra_args=args.extra_args,
        ):
            ok += 1

    total = len(queries)
    print(f"Smoke test completed: {ok}/{total} passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
