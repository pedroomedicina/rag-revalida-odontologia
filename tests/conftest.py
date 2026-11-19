from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.fixtures.corpus_builder import build_sample_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "configs" / "metadata_template.json"
STATUS_DOC = "docs/status_2026-11-19.md"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """`known_bug` tests describe the correct behavior and fail until the bug is fixed.

    They run as strict xfail: once a fix lands they XPASS, which fails the suite,
    so the marker has to be removed in the same change as the fix.
    """
    for item in items:
        marker = item.get_closest_marker("known_bug")
        if marker is not None:
            reason = marker.args[0] if marker.args else "known bug"
            item.add_marker(pytest.mark.xfail(strict=True, reason=f"{reason} ({STATUS_DOC})"))


@dataclass
class PrepRun:
    corpus: Path
    clean: Path
    out: Path
    returncode: int
    stdout: str

    @property
    def inventory(self) -> list[dict[str, str]]:
        with (self.out / "inventory.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @property
    def chunks(self) -> list[dict[str, object]]:
        lines = (self.out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def report(self, name: str) -> dict[str, object]:
        return json.loads((self.out / f"{name}.json").read_text(encoding="utf-8"))

    @property
    def cleanup_report(self) -> dict[str, object]:
        return json.loads((self.clean / "cleanup_report.json").read_text(encoding="utf-8"))

    def normalized(self, doc_id: str) -> dict[str, object]:
        path = self.out / "normalized" / f"{doc_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def row_for(self, filename_fragment: str) -> dict[str, str]:
        matches = [r for r in self.inventory if filename_fragment in r["filename"]]
        assert len(matches) == 1, (filename_fragment, [r["filename"] for r in self.inventory])
        return matches[0]


@pytest.fixture(scope="session")
def sample_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_sample_corpus(tmp_path_factory.mktemp("sample"))


@pytest.fixture(scope="session")
def prep_run(sample_corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> PrepRun:
    """Run the full prep pipeline once on the sample corpus."""
    out = tmp_path_factory.mktemp("prep_out")
    clean = sample_corpus.parent / "corpus_clean"
    cmd = [
        sys.executable,
        "-m",
        "rag.run_mvp_prep",
        str(sample_corpus),
        "--template",
        str(TEMPLATE),
        "--clean-output",
        str(clean),
        "--inventory-csv",
        str(out / "inventory.csv"),
        "--validation-report",
        str(out / "validation.json"),
        "--ocr-m1-report",
        str(out / "ocr_m1.json"),
        "--ocr-m2-report",
        str(out / "ocr_m2.json"),
        "--normalized-output-dir",
        str(out / "normalized"),
        "--ingestion-report",
        str(out / "ingestion.json"),
        "--chunks-output",
        str(out / "chunks.jsonl"),
        "--chunking-report",
        str(out / "chunking.json"),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return PrepRun(
        corpus=sample_corpus,
        clean=clean,
        out=out,
        returncode=result.returncode,
        stdout=result.stdout + result.stderr,
    )
