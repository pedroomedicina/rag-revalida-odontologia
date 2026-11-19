"""Citation validation in the answer layer."""

from __future__ import annotations

import pytest

from rag.answer_with_rag import EvidenceItem, build_user_prompt, safe_abstention, validate_citations

EVIDENCE = [
    EvidenceItem("E1", "D-ch-0001", "[D p.2]", "Guía", "", "texto uno", 1.0),
    EvidenceItem("E2", "D-ch-0002", "[D pp.3-4]", "Guía", "", "texto dos", 0.9),
]


def source(evidence_id: str, citation: str) -> dict[str, str]:
    return {"evidence_id": evidence_id, "citation": citation, "motivo": "x"}


def test_valid_citations_pass():
    payload = {"fuentes": [source("E1", "[D p.2]"), source("E2", "[D pp.3-4]")]}
    assert validate_citations(payload, EVIDENCE) == []


def test_wrong_page_is_rejected():
    errors = validate_citations({"fuentes": [source("E1", "[D p.3]")]}, EVIDENCE)
    assert errors and "citation mismatch" in errors[0]


def test_unknown_evidence_is_rejected():
    errors = validate_citations({"fuentes": [source("E9", "[D p.2]")]}, EVIDENCE)
    assert errors and "unknown evidence_id" in errors[0]


def test_fuentes_must_be_a_list():
    assert validate_citations({"fuentes": "E1"}, EVIDENCE) == ["Field 'fuentes' must be a list."]


def test_abstention_has_no_sources():
    out = safe_abstention("¿?", ["err"])
    assert out["suficiencia_evidencia"] == "insuficiente"
    assert out["fuentes"] == []


def test_prompt_includes_every_evidence_citation():
    prompt = build_user_prompt("¿Qué?", EVIDENCE)
    assert "citation: [D p.2]" in prompt and "citation: [D pp.3-4]" in prompt


@pytest.mark.known_bug("a 'suficiente' answer with no sources passes validation")
def test_sufficient_answer_requires_sources():
    payload = {"suficiencia_evidencia": "suficiente", "fuentes": [], "respuesta": "Sí."}
    assert validate_citations(payload, EVIDENCE) != []


@pytest.mark.known_bug("citations in fragmentos_de_apoyo are not validated")
def test_fragment_citations_are_validated():
    payload = {
        "fuentes": [source("E1", "[D p.2]")],
        "fragmentos_de_apoyo": [{"evidence_id": "E9", "citation": "[X p.1]", "fragmento": "?"}],
    }
    assert validate_citations(payload, EVIDENCE) != []
