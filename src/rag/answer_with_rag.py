#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = """Eres un asistente académico en español para estudio de odontología.
Responde SOLO con base en la evidencia entregada.
No inventes datos.
Si la evidencia es insuficiente o conflictiva, dilo explícitamente.
Todas las afirmaciones relevantes deben poder trazarse a citas entregadas.
"""


OUTPUT_SCHEMA_HINT = """Devuelve JSON válido con esta estructura exacta:
{
  "respuesta": "string",
  "suficiencia_evidencia": "suficiente|insuficiente",
  "fuentes": [
    {
      "evidence_id": "E1",
      "citation": "[DOC_ID p.X] o [DOC_ID pp.X-Y]",
      "motivo": "por qué esta fuente respalda la respuesta"
    }
  ],
  "fragmentos_de_apoyo": [
    {
      "evidence_id": "E1",
      "citation": "[DOC_ID p.X] o [DOC_ID pp.X-Y]",
      "fragmento": "cita breve literal o parafraseada muy cercana al texto"
    }
  ],
  "observaciones": ["string"]
}
"""


@dataclass
class EvidenceItem:
    evidence_id: str
    chunk_id: str
    citation: str
    title: str
    section_title: str
    text: str
    final_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG answering using hybrid retrieval + OpenAI generation."
    )
    parser.add_argument("--query", required=True, help="User question in Spanish")
    parser.add_argument(
        "--retrieve-script",
        type=Path,
        default=Path("scripts/retrieve_hybrid.py"),
        help="Path to retrieve_hybrid.py script",
    )
    parser.add_argument("--dsn", default=os.getenv("DATABASE_URL", ""), help="Postgres DSN")
    parser.add_argument("--table", default="rag_chunks", help="Vector/FTS table")
    parser.add_argument("--k-final", type=int, default=8, help="Evidence chunks for generator")
    parser.add_argument("--k-vector", type=int, default=30)
    parser.add_argument("--k-keyword", type=int, default=30)
    parser.add_argument("--k-merge", type=int, default=40)
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-base",
        help="Embedding model name (must match index side)",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        help="Cross-encoder model name",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1-mini",
        help="OpenAI model for final answer generation",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY)",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument(
        "--strict-citations",
        action="store_true",
        help="Fail-safe to abstention if model emits invalid citations",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include retrieval debug payload in output",
    )
    return parser


def run_retrieval(args: argparse.Namespace) -> dict[str, object]:
    retrieve_script = args.retrieve_script.expanduser().resolve()
    if not retrieve_script.exists():
        raise RuntimeError(f"retrieve script not found: {retrieve_script}")
    if not args.dsn:
        raise RuntimeError("Missing --dsn and DATABASE_URL is not set.")

    cmd = [
        sys.executable,
        str(retrieve_script),
        "--query",
        args.query,
        "--dsn",
        args.dsn,
        "--table",
        args.table,
        "--embedding-model",
        args.embedding_model,
        "--reranker-model",
        args.reranker_model,
        "--k-vector",
        str(args.k_vector),
        "--k-keyword",
        str(args.k_keyword),
        "--k-merge",
        str(args.k_merge),
        "--k-final",
        str(args.k_final),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"retrieve_hybrid failed: {stderr}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        preview = proc.stdout[:800]
        raise RuntimeError(f"retrieve_hybrid output is not valid JSON. Preview: {preview}") from exc


def build_evidence(rows: Sequence[dict[str, object]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for idx, row in enumerate(rows, start=1):
        items.append(
            EvidenceItem(
                evidence_id=f"E{idx}",
                chunk_id=str(row.get("chunk_id", "")),
                citation=str(row.get("citation", "")),
                title=str(row.get("title", "")),
                section_title=str(row.get("section_title", "")),
                text=str(row.get("text", "")),
                final_score=float(row.get("final_score", 0.0) or 0.0),
            )
        )
    return items


def build_user_prompt(query: str, evidence_items: Sequence[EvidenceItem]) -> str:
    lines = [
        "Pregunta del usuario:",
        query,
        "",
        "Evidencia disponible (usar solo esto):",
    ]
    for item in evidence_items:
        lines.append(f"- evidence_id: {item.evidence_id}")
        lines.append(f"  citation: {item.citation}")
        if item.title.strip():
            lines.append(f"  title: {item.title.strip()}")
        if item.section_title.strip():
            lines.append(f"  section_title: {item.section_title.strip()}")
        lines.append(f"  text: {item.text.strip()}")
        lines.append("")
    lines.append("Reglas de respuesta:")
    lines.append("- Responde en español.")
    lines.append("- No uses conocimiento externo.")
    lines.append("- Cita solo citations de la evidencia entregada.")
    lines.append("- Si falta respaldo, marca suficiencia_evidencia='insuficiente'.")
    lines.append("")
    lines.append(OUTPUT_SCHEMA_HINT)
    return "\n".join(lines)


def generate_answer_openai(
    query: str,
    evidence_items: Sequence[EvidenceItem],
    model: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, object]:
    if not api_key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --openai-api-key.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install OpenAI SDK first: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    prompt = build_user_prompt(query=query, evidence_items=evidence_items)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Model did not return valid JSON. Raw output: {content[:1200]}"
        ) from exc


SUFFICIENCY_VALUES = ("suficiente", "insuficiente")

# A bracketed page reference written in the answer text: "[DOC p.3]", "[DOC pp.3-5]",
# and loose variants such as "[DOC, p. 3]".
INLINE_CITATION_RE = re.compile(r"\[[^\[\]]*?\bpp?\.\s*\d+(?:\s*[-–]\s*\d+)?\s*\]")


def _citation_key(citation: str) -> str:
    """Compare citations ignoring spaces, commas, and dash style."""
    return re.sub(r"[\s,]", "", citation).replace("–", "-")


def _check_cited_items(
    field: str,
    items: object,
    allowed_by_id: dict[str, str],
    errors: list[str],
) -> int:
    """Validate a list of {evidence_id, citation} objects; return how many were valid."""
    if items is None:
        return 0
    if not isinstance(items, list):
        errors.append(f"Field '{field}' must be a list.")
        return 0
    valid = 0
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"{field}[{idx}] must be an object.")
            continue
        evidence_id = str(item.get("evidence_id", "")).strip()
        citation = str(item.get("citation", "")).strip()
        expected = allowed_by_id.get(evidence_id)
        if expected is None:
            errors.append(f"{field}[{idx}] uses unknown evidence_id '{evidence_id}'.")
            continue
        if citation != expected:
            errors.append(
                f"{field}[{idx}] citation mismatch for {evidence_id}: "
                f"got '{citation}' expected '{expected}'."
            )
            continue
        valid += 1
    return valid


def validate_citations(
    payload: dict[str, object], evidence_items: Sequence[EvidenceItem]
) -> list[str]:
    """Check that every citation in the model's answer points at retrieved evidence.

    - `suficiencia_evidencia` is "suficiente" or "insuficiente"; a "suficiente" answer
      must cite at least one valid source.
    - Every entry in `fuentes` and `fragmentos_de_apoyo` names a retrieved evidence_id and
      repeats that chunk's citation exactly.
    - Every bracketed page citation written inside `respuesta` is one of the retrieved
      chunks' citations.

    Returns a list of errors; empty means valid. Paraphrased fragments are allowed, so
    fragment text is not compared with the chunk text.
    """
    allowed_by_id = {item.evidence_id: item.citation for item in evidence_items}
    errors: list[str] = []

    sufficiency = str(payload.get("suficiencia_evidencia", "")).strip()
    if sufficiency not in SUFFICIENCY_VALUES:
        errors.append(
            f"Field 'suficiencia_evidencia' must be one of {SUFFICIENCY_VALUES}, "
            f"got '{sufficiency}'."
        )

    valid_sources = _check_cited_items("fuentes", payload.get("fuentes", []), allowed_by_id, errors)
    _check_cited_items(
        "fragmentos_de_apoyo", payload.get("fragmentos_de_apoyo"), allowed_by_id, errors
    )

    if sufficiency == "suficiente" and valid_sources == 0:
        errors.append("A 'suficiente' answer must cite at least one retrieved source in 'fuentes'.")

    allowed_keys = {_citation_key(citation) for citation in allowed_by_id.values()}
    respuesta = str(payload.get("respuesta", ""))
    for inline in INLINE_CITATION_RE.findall(respuesta):
        if _citation_key(inline) not in allowed_keys:
            errors.append(f"'respuesta' cites {inline}, which is not a retrieved chunk.")
    return errors


def safe_abstention(query: str, validation_errors: Sequence[str]) -> dict[str, object]:
    notes = [
        "No pude garantizar citas válidas con la evidencia recuperada.",
        "Se recomienda revisar la consulta o reintentar con más contexto.",
    ]
    notes.extend(validation_errors[:5])
    return {
        "query": query,
        "respuesta": "No cuento con respaldo suficiente y verificable para responder con seguridad usando solo el corpus recuperado.",
        "suficiencia_evidencia": "insuficiente",
        "fuentes": [],
        "fragmentos_de_apoyo": [],
        "observaciones": notes,
    }


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    if not args.query.strip():
        raise SystemExit("Query cannot be empty.")

    retrieval_payload = run_retrieval(args)
    rows = retrieval_payload.get("results", [])
    if not isinstance(rows, list) or not rows:
        output = {
            "query": args.query,
            "respuesta": "No encontré evidencia suficiente en el corpus para responder esta pregunta.",
            "suficiencia_evidencia": "insuficiente",
            "fuentes": [],
            "fragmentos_de_apoyo": [],
            "observaciones": ["Retrieval no devolvió chunks relevantes."],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    evidence_items = build_evidence(rows)
    generated = generate_answer_openai(
        query=args.query,
        evidence_items=evidence_items,
        model=args.openai_model,
        api_key=args.openai_api_key,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    citation_errors = validate_citations(generated, evidence_items)
    if citation_errors and args.strict_citations:
        output = safe_abstention(args.query, citation_errors)
    else:
        output = {
            "query": args.query,
            **generated,
        }
        if citation_errors:
            observations = output.get("observaciones", [])
            if not isinstance(observations, list):
                observations = []
            observations.append("Advertencia de validación de citas.")
            observations.extend(citation_errors[:5])
            output["observaciones"] = observations

    if args.debug:
        output["debug"] = {
            "retrieval": retrieval_payload.get("retrieval", {}),
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "chunk_id": item.chunk_id,
                    "citation": item.citation,
                    "final_score": item.final_score,
                }
                for item in evidence_items
            ],
            "citation_errors": citation_errors,
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
