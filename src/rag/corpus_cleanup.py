#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

JUNK_NAME_RE = re.compile(r"(^\._)|(^\.ds_store$)|(^thumbs\.db$)|(~$)|(^~\$)", re.IGNORECASE)
# Not \b: "_" is a word character, so \bcopia\b misses "guia_copia".
DUP_HINT_RE = re.compile(r"\(\d+\)|(?<![^\W_])(?:copy|copia)(?![^\W_])", re.IGNORECASE)
NON_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")
PDF_PAGE_RE = re.compile(rb"/Type\s*/Page\b")
TITLE_SEPARATORS_RE = re.compile(r"[._-]+")
# Applied to keys whose only separator is "_": "copia_de_x" / "x_copia".
COPY_PREFIX_RE = re.compile(r"^(?:copia_de|copy_of)_")
COPY_WORD_RE = re.compile(r"(?:^|_)(?:copy|copia)(?=_|$)")
TRAILING_NUMBER_RE = re.compile(r"_\d+$")


@dataclass(frozen=True)
class FileEntry:
    path: Path
    size: int
    ext: str
    is_junk: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_normalized_filename(path: Path) -> str:
    stem = unicodedata.normalize("NFKD", path.stem)
    stem = "".join(ch for ch in stem if not unicodedata.combining(ch))
    stem = stem.replace("&", " and ")
    stem = stem.strip().lower()
    stem = NON_SAFE_CHARS_RE.sub("_", stem).strip("._-")
    stem = re.sub(r"_+", "_", stem)
    if not stem:
        stem = "file"
    ext = path.suffix.lower()
    return f"{stem}{ext}"


def normalized_title_key(path: Path) -> str:
    stem = unicodedata.normalize("NFKD", path.stem)
    stem = "".join(ch for ch in stem if not unicodedata.combining(ch))
    stem = stem.lower()
    stem = stem.replace("&", " and ")
    stem = NON_SAFE_CHARS_RE.sub("_", stem)
    # One separator, so "Guía - X", "Guia_X" and "guia.x" share a key.
    stem = TITLE_SEPARATORS_RE.sub("_", stem).strip("_")
    # Common "copy-ish" markers from exported/resaved files.
    stem = COPY_PREFIX_RE.sub("", stem)
    stem = COPY_WORD_RE.sub("", stem).strip("_")
    stem = TRAILING_NUMBER_RE.sub("", stem)
    # Common conversion artifacts in names, e.g. ".docx.pdf".
    stem = re.sub(r"_(docx|pptx|xlsx|txt)$", "", stem)
    return stem or "file"


def estimate_pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    return len(PDF_PAGE_RE.findall(data))


def detect_near_duplicate_pdf_resaves(
    filtered_files: list[FileEntry],
    hash_to_paths: dict[str, list[Path]],
    max_size_delta_pct: float,
    max_page_delta: int,
) -> list[dict[str, object]]:
    pdf_entries = [f for f in filtered_files if f.ext == "pdf"]
    if len(pdf_entries) < 2:
        return []

    path_to_hash: dict[Path, str] = {}
    for hash_value, paths in hash_to_paths.items():
        for p in paths:
            path_to_hash[p] = hash_value

    by_title: dict[str, list[FileEntry]] = defaultdict(list)
    for f in pdf_entries:
        by_title[normalized_title_key(f.path)].append(f)

    page_counts: dict[Path, int] = {}
    pairs: list[dict[str, object]] = []
    seen = set()
    for title_key, group in by_title.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda x: x.path.name.lower())
        for i in range(len(group_sorted)):
            for j in range(i + 1, len(group_sorted)):
                a = group_sorted[i]
                b = group_sorted[j]
                if path_to_hash.get(a.path) == path_to_hash.get(b.path):
                    # Exact duplicates are already handled by hash dedupe.
                    continue

                max_size = max(a.size, b.size)
                if max_size == 0:
                    continue
                delta_pct = abs(a.size - b.size) / max_size * 100.0
                if delta_pct > max_size_delta_pct:
                    continue

                if a.path not in page_counts:
                    page_counts[a.path] = estimate_pdf_page_count(a.path)
                if b.path not in page_counts:
                    page_counts[b.path] = estimate_pdf_page_count(b.path)
                page_delta = abs(page_counts[a.path] - page_counts[b.path])
                if page_delta > max_page_delta:
                    continue

                key = tuple(sorted((str(a.path), str(b.path))))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "title_key": title_key,
                        "file_a": str(a.path),
                        "file_b": str(b.path),
                        "size_a": a.size,
                        "size_b": b.size,
                        "size_delta_pct": round(delta_pct, 3),
                        "pages_a": page_counts[a.path],
                        "pages_b": page_counts[b.path],
                        "page_delta": page_delta,
                    }
                )

    return sorted(pairs, key=lambda p: (p["title_key"], p["size_delta_pct"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten and clean a corpus safely (dry-run by default)."
    )
    parser.add_argument("source", type=Path, help="Source corpus directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for cleaned corpus (default: <source>_clean)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview actions only (default behavior).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform copy/write actions to output folder.",
    )
    parser.add_argument(
        "--remove-junk",
        action="store_true",
        help="Exclude junk files (.DS_Store, ._*, Thumbs.db, temp files).",
    )
    parser.add_argument(
        "--dedupe-hash",
        action="store_true",
        help="Keep one file per exact-content hash.",
    )
    parser.add_argument(
        "--normalize-names",
        action="store_true",
        help="Normalize output names to ascii-safe snake_case.",
    )
    parser.add_argument(
        "--keep-extensions",
        default="pdf,docx,xlsx",
        help="Comma-separated extension allowlist (without dots).",
    )
    parser.add_argument(
        "--pdf-doc-only",
        action="store_true",
        help="Shortcut for keeping only PDF + Word variants (pdf, doc, docx, docm, dot, dotx).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path to write JSON report (default: <output>/cleanup_report.json in --apply, ./cleanup_report.json in dry-run).",
    )
    parser.add_argument(
        "--near-dup-pdf-resave-report",
        action="store_true",
        help="Report likely PDF export/resave variants (same normalized title + close size + close page count).",
    )
    parser.add_argument(
        "--near-dup-size-delta-pct",
        type=float,
        default=12.0,
        help="Maximum size delta percentage for near-duplicate PDF pair detection.",
    )
    parser.add_argument(
        "--near-dup-page-delta",
        type=int,
        default=0,
        help="Maximum page-count delta for near-duplicate PDF pair detection.",
    )
    return parser


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def choose_canonical(paths: list[Path]) -> Path:
    def score(p: Path) -> tuple[int, int, int, int]:
        name = p.name
        return (
            1 if DUP_HINT_RE.search(name) else 0,
            p.parts.count("ENDODONCIA"),
            len(name),
            len(str(p)),
        )

    return sorted(paths, key=score)[0]


def ensure_unique_name(target_dir: Path, candidate_name: str) -> Path:
    out = target_dir / candidate_name
    if not out.exists():
        return out
    stem = out.stem
    ext = out.suffix
    i = 2
    while True:
        alt = target_dir / f"{stem}__{i}{ext}"
        if not alt.exists():
            return alt
        i += 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    output = (args.output or source.parent / f"{source.name}_clean").expanduser().resolve()
    apply_mode = bool(args.apply)
    dry_run = not apply_mode
    keep_exts = {
        ext.strip().lower().lstrip(".") for ext in args.keep_extensions.split(",") if ext.strip()
    }
    if args.pdf_doc_only:
        keep_exts = {"pdf", "doc", "docx", "docm", "dot", "dotx"}

    scanned_files: list[FileEntry] = []
    for path in iter_files(source):
        ext = path.suffix.lower().lstrip(".")
        stat = path.stat()
        scanned_files.append(
            FileEntry(
                path=path,
                size=stat.st_size,
                ext=ext if ext else "<no_extension>",
                is_junk=bool(JUNK_NAME_RE.search(path.name)),
            )
        )

    ext_counter = Counter(f.ext for f in scanned_files)
    total_bytes = sum(f.size for f in scanned_files)

    # Filter by allowlist and junk strategy
    filtered = []
    dropped_junk = []
    dropped_ext = []
    for f in scanned_files:
        if args.remove_junk and f.is_junk:
            dropped_junk.append(f.path)
            continue
        if f.ext == "<no_extension>" or f.ext not in keep_exts:
            dropped_ext.append(f.path)
            continue
        filtered.append(f)

    # Hash groups
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for f in filtered:
        hash_to_paths[sha256_file(f.path)].append(f.path)

    duplicate_groups = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    duplicate_extra_copies = sum(len(paths) - 1 for paths in duplicate_groups)
    reclaimable_bytes = sum(
        (len(paths) - 1) * paths[0].stat().st_size for paths in duplicate_groups
    )
    near_dup_pdf_pairs: list[dict[str, object]] = []
    if args.near_dup_pdf_resave_report:
        near_dup_pdf_pairs = detect_near_duplicate_pdf_resaves(
            filtered_files=filtered,
            hash_to_paths=hash_to_paths,
            max_size_delta_pct=args.near_dup_size_delta_pct,
            max_page_delta=args.near_dup_page_delta,
        )

    selected_paths: list[Path] = []
    if args.dedupe_hash:
        for paths in hash_to_paths.values():
            selected_paths.append(choose_canonical(paths))
    else:
        selected_paths = [f.path for f in filtered]

    # Plan output names
    planned_outputs: list[dict[str, str]] = []
    name_usage: dict[str, int] = defaultdict(int)
    for src_path in sorted(selected_paths):
        desired_name = safe_normalized_filename(src_path) if args.normalize_names else src_path.name
        name_usage[desired_name] += 1
        if name_usage[desired_name] == 1:
            final_name = desired_name
        else:
            stem = Path(desired_name).stem
            ext = Path(desired_name).suffix
            final_name = f"{stem}__{name_usage[desired_name]}{ext}"
        planned_outputs.append({"source": str(src_path), "dest_name": final_name})

    # Execute
    copied = 0
    if apply_mode:
        output.mkdir(parents=True, exist_ok=True)
        for item in planned_outputs:
            src = Path(item["source"])
            target = ensure_unique_name(output, item["dest_name"])
            shutil.copy2(src, target)
            copied += 1

    report = {
        "source": str(source),
        "output": str(output),
        "mode": "apply" if apply_mode else "dry-run",
        "totals": {
            "scanned_files": len(scanned_files),
            "scanned_total_mb": round(total_bytes / 1024 / 1024, 2),
            "filtered_files": len(filtered),
            "selected_for_output": len(selected_paths),
            "copied_files": copied,
        },
        "extension_counts": dict(sorted(ext_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "drops": {
            "junk_count": len(dropped_junk),
            "by_extension_count": len(dropped_ext),
        },
        "dedupe": {
            "hash_groups_total": len(hash_to_paths),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_extra_copies": duplicate_extra_copies,
            "reclaimable_mb_if_deduped": round(reclaimable_bytes / 1024 / 1024, 2),
        },
        "near_duplicate_pdf_resave": {
            "enabled": bool(args.near_dup_pdf_resave_report),
            "size_delta_pct_threshold": args.near_dup_size_delta_pct,
            "page_delta_threshold": args.near_dup_page_delta,
            "candidate_pairs_count": len(near_dup_pdf_pairs),
            "candidate_pairs_preview": near_dup_pdf_pairs[:40],
        },
        "sample_duplicate_groups": [[str(p) for p in paths[:5]] for paths in duplicate_groups[:10]],
        "planned_outputs_preview": planned_outputs[:30],
    }

    report_path = args.report_json
    if report_path is None:
        if apply_mode:
            report_path = output / "cleanup_report.json"
        else:
            report_path = Path.cwd() / "cleanup_report.json"
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Mode: {'APPLY' if apply_mode else 'DRY-RUN'}")
    print(f"Scanned files: {len(scanned_files)}")
    print(f"Filtered files: {len(filtered)}")
    print(f"Selected for output: {len(selected_paths)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Extra duplicate copies: {duplicate_extra_copies}")
    print(f"Reclaimable by exact dedupe: {round(reclaimable_bytes / 1024 / 1024, 2)} MB")
    if args.near_dup_pdf_resave_report:
        print(f"Near-duplicate PDF resave candidate pairs: {len(near_dup_pdf_pairs)}")
    print(f"Report: {report_path}")
    if dry_run:
        print("No files were copied. Re-run with --apply to execute.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
