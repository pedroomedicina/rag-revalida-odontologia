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


def fragment(evidence_id: str, citation: str) -> dict[str, str]:
    return {"evidence_id": evidence_id, "citation": citation, "fragmento": "texto"}


def answer(**overrides: object) -> dict[str, object]:
    """A valid 'suficiente' answer; each test changes one thing."""
    payload: dict[str, object] = {
        "respuesta": "El hidróxido de calcio se usa entre sesiones [D p.2].",
        "suficiencia_evidencia": "suficiente",
        "fuentes": [source("E1", "[D p.2]")],
        "fragmentos_de_apoyo": [fragment("E1", "[D p.2]")],
        "observaciones": [],
    }
    payload.update(overrides)
    return payload


def errors_for(**overrides: object) -> list[str]:
    return validate_citations(answer(**overrides), EVIDENCE)


def test_valid_answer_passes():
    assert errors_for() == []


def test_valid_answer_with_page_range_passes():
    assert (
        errors_for(
            respuesta="Ver [D pp.3-4] y [D p.2].",
            fuentes=[source("E1", "[D p.2]"), source("E2", "[D pp.3-4]")],
        )
        == []
    )


def test_insufficient_answer_without_sources_passes():
    assert (
        errors_for(
            respuesta="No encontré respaldo.",
            suficiencia_evidencia="insuficiente",
            fuentes=[],
            fragmentos_de_apoyo=[],
        )
        == []
    )


@pytest.mark.parametrize("value", ["", "parcial", None])
def test_sufficiency_must_be_a_known_value(value):
    errors = errors_for(suficiencia_evidencia=value)
    assert any("suficiencia_evidencia" in e for e in errors)


def test_wrong_page_is_rejected():
    errors = errors_for(fuentes=[source("E1", "[D p.3]")])
    assert any("citation mismatch" in e for e in errors)


def test_unknown_evidence_is_rejected():
    errors = errors_for(fuentes=[source("E9", "[D p.2]")])
    assert any("fuentes[1] uses unknown evidence_id 'E9'" in e for e in errors)


def test_fuentes_must_be_a_list():
    assert "Field 'fuentes' must be a list." in errors_for(fuentes="E1")


def test_sufficient_answer_requires_sources():
    errors = errors_for(fuentes=[])
    assert any("must cite at least one retrieved source" in e for e in errors)


def test_sufficient_answer_with_only_invalid_sources_is_rejected():
    errors = errors_for(fuentes=[source("E9", "[X p.1]")])
    assert any("must cite at least one retrieved source" in e for e in errors)


def test_fragment_citations_are_validated():
    errors = errors_for(fragmentos_de_apoyo=[fragment("E9", "[X p.1]")])
    assert any("fragmentos_de_apoyo[1] uses unknown evidence_id" in e for e in errors)


def test_fragment_with_wrong_page_is_rejected():
    errors = errors_for(fragmentos_de_apoyo=[fragment("E2", "[D p.3]")])
    assert any("fragmentos_de_apoyo[1] citation mismatch" in e for e in errors)


def test_inline_citation_to_unretrieved_page_is_rejected():
    errors = errors_for(respuesta="Se usa entre sesiones [D p.9].")
    assert errors == ["'respuesta' cites [D p.9], which is not a retrieved chunk."]


def test_inline_citation_to_other_document_is_rejected():
    errors = errors_for(respuesta="Según [OTRO pp.3-4], sí.")
    assert len(errors) == 1 and "[OTRO pp.3-4]" in errors[0]


def test_inline_citation_spacing_variants_are_accepted():
    assert errors_for(respuesta="Ver [D, p. 2] y [D pp. 3 - 4].") == []


def test_non_citation_brackets_are_ignored():
    assert errors_for(respuesta="Concentración [0,5% a 5,25%] según [D p.2].") == []


def test_abstention_has_no_sources():
    out = safe_abstention("¿?", ["err"])
    assert out["suficiencia_evidencia"] == "insuficiente"
    assert out["fuentes"] == []
    assert validate_citations(out, EVIDENCE) == []


def test_prompt_includes_every_evidence_citation():
    prompt = build_user_prompt("¿Qué?", EVIDENCE)
    assert "citation: [D p.2]" in prompt and "citation: [D pp.3-4]" in prompt
