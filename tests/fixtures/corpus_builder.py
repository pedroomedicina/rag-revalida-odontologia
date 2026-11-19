"""Build a small synthetic Spanish corpus that exercises the prep pipeline.

Pure Python (no textutil/sips/LibreOffice) so it runs on Linux CI. The text is written
for these tests; it is not Universidad de Chile course material.
"""

from __future__ import annotations

import shutil
import textwrap
import zipfile
from pathlib import Path

GUIDE_PAGES: list[list[str]] = [
    [
        "MEDICACIÓN INTRACANAL",
        "",
        "El hidróxido de calcio se indica como medicación intracanal entre sesiones cuando "
        "existe necrosis pulpar con lesión periapical. Su pH alcalino, cercano a 12,5, genera "
        "un efecto antibacteriano y favorece la reparación de tejidos duros.",
        "",
        "Se recomienda mantener la medicación por un período de 7 a 14 días, sellando la "
        "cavidad de acceso con un material provisorio de buen sellado marginal.",
    ],
    [
        "IRRIGACIÓN",
        "",
        "El hipoclorito de sodio es el irrigante principal durante la preparación "
        "químico-mecánica. Disuelve tejido orgánico, tiene acción antimicrobiana y lubrica el "
        "conducto durante la instrumentación.",
        "",
        "Las concentraciones utilizadas varían entre 0,5% y 5,25%. Se debe irrigar con aguja "
        "de salida lateral sin trabar la aguja en el conducto.",
        "",
        "PROTOCOLO DE IRRIGACIÓN FINAL",
        "",
        "Paso 1: irrigar con hipoclorito de sodio durante toda la instrumentación del conducto.",
    ],
    [
        "Paso 2: aplicar EDTA al 17% durante un minuto para remover el barro dentinario.",
        "",
        "Paso 3: realizar la irrigación final con hipoclorito de sodio y secar con conos de "
        "papel antes de obturar.",
    ],
]

LONG_PARAGRAPH = (
    "La obturación del sistema de conductos busca un sellado tridimensional que impida la "
    "recontaminación bacteriana y permita la reparación periapical. "
) * 6

DIAGNOSIS_PARAGRAPHS = [
    "DIAGNÓSTICO PULPAR",
    "La biopulpectomía se realiza en dientes con pulpa vital, mientras que la "
    "necropulpectomía corresponde al tratamiento de dientes con pulpa necrótica.",
    "En la necropulpectomía el objetivo principal es la desinfección del sistema de conductos.",
]

# Magic bytes of an OLE2 compound file (legacy Word .doc). Enough to be "not a zip".
OLE2_HEADER = bytes.fromhex("d0cf11e0a1b11ae1") + b"\x00" * 504


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[str]], *, text_layer: bool = True) -> None:
    """Write a minimal Helvetica/WinAnsi PDF. Each page is a list of lines ('' = blank).

    With text_layer=False the pages only contain vector strokes, like a scan without OCR.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    pages_id = add(b"")  # placeholder, filled once kids are known
    kids: list[int] = []
    for lines in pages:
        if text_layer:
            wrapped: list[str] = []
            for line in lines:
                wrapped += textwrap.wrap(line, 85) or [""]
            ops = ["BT /F1 11 Tf 14 TL 60 780 Td"]
            ops += [f"({_pdf_escape(w)}) Tj T*" for w in wrapped]
            ops.append("ET")
        else:
            ops = ["0.2 G 60 700 m 550 700 l S 60 650 m 550 650 l S"]
        stream = "\n".join(ops).encode("cp1252")
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(
            add(
                (
                    f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).encode()
            )
        )
    kid_refs = " ".join(f"{k} 0 R" for k in kids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kid_refs}] /Count {len(kids)} >>".encode()
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


def write_docx(path: Path, paragraphs: list[str]) -> None:
    """Write a minimal but valid .docx with one w:p per paragraph."""
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs)
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)


def build_sample_corpus(root: Path) -> Path:
    """Create the sample corpus under root/corpus and return that directory.

    Contents (9 files, one junk):
    - a 3-page guide whose final irrigation protocol spans pp. 2-3
    - a byte-identical copy of it in a subfolder (exact dedupe)
    - a "copia" resave with one sentence changed (near-duplicate)
    - a 1-page document with one paragraph longer than the 800-token chunk max
    - a .docx, a legacy .doc, a PDF without a text layer, and .DS_Store
    - a corrupt .pdf that no extractor or repair can read (fails ingestion everywhere)
    """
    corpus = root / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    (corpus / "sub").mkdir(parents=True)

    guide = corpus / "Guía Endodoncia - Medicación.pdf"
    write_pdf(guide, GUIDE_PAGES)
    shutil.copy(guide, corpus / "sub" / "Guía Endodoncia - Medicación (1).pdf")

    revised = [list(page) for page in GUIDE_PAGES]
    revised[2][-1] += " Revisado 2024."
    write_pdf(corpus / "sub" / "Guia_Endodoncia_Medicacion_copia.pdf", revised)

    write_pdf(corpus / "Obturacion_Larga.pdf", [["OBTURACIÓN", ""] + [LONG_PARAGRAPH, ""] * 8])
    write_docx(corpus / "Apunte Diagnóstico.docx", DIAGNOSIS_PARAGRAPHS)
    (corpus / "Apunte_Legacy.doc").write_bytes(OLE2_HEADER)
    write_pdf(corpus / "Escaneo_Clase.pdf", [[""]], text_layer=False)
    (corpus / "Archivo_Danado.pdf").write_bytes(b"this is not a PDF file\n" * 20)
    (corpus / ".DS_Store").write_bytes(b"")
    return corpus
