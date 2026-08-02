#!/usr/bin/env python3
"""
Normaliza registros extraídos para un proyecto RAG del DANE.

Entrada:
    processed/extracted_records.jsonl

Salidas:
    processed/documents.jsonl
    processed/normalization_report.csv

Ejemplo:
    python -m dane_assistant.ingestion.steps.build_documents \
      --input "build/dane-ingestion/sources/documents/processed/extracted_records.jsonl" \
      --output-dir "build/dane-ingestion/sources/documents/processed"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


TOPIC_DEFAULT = "Tecnología e Innovación"
SUBTOPIC_DEFAULT = "Encuesta de Inversión en Investigación y Desarrollo I+D"

MIN_TEXT_CHARACTERS = 80
WHITESPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_whitespace(value: Any) -> str:
    """Convierte un valor en texto y elimina espacios/saltos redundantes."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def normalize_url(value: Any) -> Optional[str]:
    """Devuelve una URL limpia o None."""
    url = normalize_whitespace(value)

    if not url:
        return None

    return url


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Lee un JSONL y devuelve un dict por línea válida."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido en {path}, línea {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Registro inválido en {path}, línea {line_number}: "
                    "se esperaba un objeto JSON."
                )

            yield record


def first_nonempty(*values: Any) -> Optional[str]:
    """Devuelve el primer valor con texto útil."""
    for value in values:
        text = normalize_whitespace(value)

        if text:
            return text

    return None


def infer_year(*values: Any) -> Optional[str]:
    """Encuentra el año más reciente que aparezca en los textos entregados."""
    years = []

    for value in values:
        text = normalize_whitespace(value)

        for match in YEAR_RE.findall(text):
            years.append(int(match))

    if not years:
        return None

    return str(max(years))


def infer_document_type(
    record: Dict[str, Any],
    title: str,
    filename: str,
    source_url: Optional[str],
) -> str:
    """
    Clasifica de forma conservadora el recurso. Se puede mejorar después
    con reglas específicas basadas en títulos reales del DANE.
    """
    record_type = normalize_whitespace(record.get("record_type")).lower()

    searchable_text = " ".join(
        [
            normalize_whitespace(title).lower(),
            normalize_whitespace(filename).lower(),
            normalize_whitespace(source_url).lower(),
            record_type,
        ]
    )

    if "bolet" in searchable_text:
        return "boletin_tecnico"

    if "anexo" in searchable_text or "cuadro" in searchable_text:
        return "anexo"

    if any(
        keyword in searchable_text
        for keyword in (
            "metodolog",
            "ficha tecnica",
            "ficha_técnica",
            "manual",
            "documento tecnico",
            "documento_técnico",
        )
    ):
        return "metodologia"

    if record_type == "pdf_page":
        return "pdf"

    if record_type == "excel_sheet":
        return "anexo"

    if record_type in {"html_section", "html_article"}:
        return "html"

    return "unknown"


def infer_subtopic(title: str, source_url: Optional[str], landing_url: Optional[str]) -> Optional[str]:
    searchable_text = " ".join(
        [title.lower(), (source_url or "").lower(), (landing_url or "").lower()]
    )

    if "entic-hogares" in searchable_text or "entic hogares" in searchable_text:
        return "Encuesta TIC en Hogares (ENTIC Hogares)"
    if "entic-empresas" in searchable_text or "entic empresas" in searchable_text:
        return "Encuesta TIC en Empresas (ENTIC Empresas)"
    if "indicadores-basicos-de-tic-en-hogares" in searchable_text:
        return "Indicadores básicos TIC en Hogares"
    if "indicadores-basicos-de-tic-en-empresas" in searchable_text:
        return "Indicadores básicos TIC en Empresas"
    if "desarrollo-e-innovacion-tecnologica-edit" in searchable_text or "edits" in searchable_text:
        return "Encuesta de Desarrollo e Innovación Tecnológica (EDIT/EDITS)"
    if "encuesta-de-inversion-en-investigacion-y-desarrollo" in searchable_text:
        return "Encuesta de Inversión en Investigación y Desarrollo I+D"

    return None


def infer_extraction_method(record_type: str) -> str:
    """Asigna un nombre claro al extractor utilizado."""
    if record_type == "pdf_page":
        return "pymupdf"

    if record_type == "excel_sheet":
        return "pandas_openpyxl_or_xlrd"

    if record_type in {"html_section", "html_article"}:
        return "beautifulsoup"

    return "unknown"


def assess_extraction_quality(
    record: Dict[str, Any],
    text: str,
    record_type: str,
) -> str:
    """
    Etiqueta orientativa para revisión humana. No es una métrica de calidad
    estadística ni reemplaza validación manual.
    """
    if len(text) < MIN_TEXT_CHARACTERS:
        return "low"

    if record_type == "excel_sheet":
        return "needs_review"

    if record_type == "pdf_page":
        return "high"

    return "needs_review"


def normalize_location(record: Dict[str, Any]) -> Dict[str, Optional[str | int]]:
    """Convierte la localización del extractor a un formato único."""
    provenance = record.get("provenance") or {}
    structure = record.get("structure") or {}

    page = provenance.get("page")
    sheet = provenance.get("sheet")

    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None

    sheet = normalize_whitespace(sheet) or None

    return {
        "page": page,
        "sheet": sheet,
        "table_name": sheet if record.get("record_type") == "excel_sheet" else None,
        "row_count": structure.get("row_count"),
    }


def normalize_record(
    record: Dict[str, Any],
    *,
    topic: str,
    subtopic: str,
    min_text_characters: int,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Convierte un registro del extractor en un documento normalizado.

    Retorna:
      (documento_normalizado_o_None, fila_de_reporte)
    """
    record_id = normalize_whitespace(record.get("record_id"))
    record_type = normalize_whitespace(record.get("record_type"))

    provenance = record.get("provenance") or {}
    raw_text = record.get("text")
    text = normalize_whitespace(raw_text)

    filename = first_nonempty(
        provenance.get("filename"),
        Path(str(provenance.get("local_path") or "")).name,
        "unknown_file",
    ) or "unknown_file"

    title = first_nonempty(
        provenance.get("title"),
        filename,
        "Documento sin título",
    ) or "Documento sin título"

    asset_sha256 = first_nonempty(provenance.get("sha256"), "") or ""
    source_url = normalize_url(provenance.get("url"))
    landing_url = normalize_url(provenance.get("source_url"))

    reference_period = first_nonempty(
        provenance.get("inferred_year"),
        infer_year(title, filename, source_url, landing_url),
    )

    document_type = infer_document_type(
        record=record,
        title=title,
        filename=filename,
        source_url=source_url,
    )
    inferred_subtopic = infer_subtopic(title, source_url, landing_url) or subtopic

    extraction_method = infer_extraction_method(record_type)
    extraction_quality = assess_extraction_quality(record, text, record_type)
    location = normalize_location(record)

    report_base = {
        "record_id": record_id,
        "record_type": record_type,
        "asset_filename": filename,
        "publication_title": title,
        "document_type": document_type,
        "reference_period": reference_period or "",
        "source_url": source_url or "",
        "landing_url": landing_url or "",
        "text_characters": len(text),
        "extraction_quality": extraction_quality,
        "status": "",
        "reason": "",
    }

    if record_type in {"excel_sheet", "csv_table"}:
        report_base["status"] = "routed_to_tabular"
        report_base["reason"] = "tabular_record"
        return None, report_base

    if not record_id:
        report_base["status"] = "excluded"
        report_base["reason"] = "missing_record_id"
        return None, report_base

    if not text:
        report_base["status"] = "excluded"
        report_base["reason"] = "empty_text"
        return None, report_base

    if len(text) < min_text_characters:
        report_base["status"] = "excluded"
        report_base["reason"] = f"text_shorter_than_{min_text_characters}_characters"
        return None, report_base

    if not asset_sha256:
        report_base["status"] = "included_with_warning"
        report_base["reason"] = "missing_asset_sha256"
    elif not source_url:
        report_base["status"] = "included_with_warning"
        report_base["reason"] = "missing_source_url"
    else:
        report_base["status"] = "included"
        report_base["reason"] = ""

    document = {
        "document_id": record_id,
        "topic": topic,
        "subtopic": inferred_subtopic,
        "publication_title": title,
        "document_type": document_type,
        "reference_period": reference_period,
        "published_date": provenance.get("inferred_date") or None,
        "source_url": source_url,
        "landing_url": landing_url,
        "asset_sha256": asset_sha256 or None,
        "asset_filename": filename,
        "location": location,
        "text": text,
        "extraction_method": extraction_method,
        "extraction_quality": extraction_quality,
        "created_at": utc_now(),
    }

    return document, report_base


def write_jsonl(documents: list[Dict[str, Any]], output_path: Path) -> None:
    """Escribe documentos como JSONL UTF-8."""
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False))
            handle.write("\n")


def write_report(rows: list[Dict[str, Any]], output_path: Path) -> None:
    """Escribe el reporte de normalización."""
    fieldnames = [
        "record_id",
        "record_type",
        "asset_filename",
        "publication_title",
        "document_type",
        "reference_period",
        "source_url",
        "landing_url",
        "text_characters",
        "extraction_quality",
        "status",
        "reason",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza extracted_records.jsonl en documents.jsonl."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta a extracted_records.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se crearán documents.jsonl y el reporte.",
    )
    parser.add_argument(
        "--topic",
        default=TOPIC_DEFAULT,
        help=f"Tema principal. Por defecto: {TOPIC_DEFAULT}",
    )
    parser.add_argument(
        "--subtopic",
        default=SUBTOPIC_DEFAULT,
        help=f"Subtema. Por defecto: {SUBTOPIC_DEFAULT}",
    )
    parser.add_argument(
        "--min-text-characters",
        type=int,
        default=MIN_TEXT_CHARACTERS,
        help=(
            "Mínimo de caracteres para incluir un registro. "
            f"Por defecto: {MIN_TEXT_CHARACTERS}."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: no existe el archivo de entrada: {input_path}", file=sys.stderr)
        return 2

    if args.min_text_characters < 1:
        print("ERROR: --min-text-characters debe ser al menos 1.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    documents_output = output_dir / "documents.jsonl"
    report_output = output_dir / "normalization_report.csv"

    documents: list[Dict[str, Any]] = []
    report_rows: list[Dict[str, Any]] = []

    try:
        for record in read_jsonl(input_path):
            document, report_row = normalize_record(
                record,
                topic=args.topic,
                subtopic=args.subtopic,
                min_text_characters=args.min_text_characters,
            )

            report_rows.append(report_row)

            if document is not None:
                documents.append(document)

    except Exception as error:
        print(f"ERROR normalizando registros: {error}", file=sys.stderr)
        return 1

    write_jsonl(documents, documents_output)
    write_report(report_rows, report_output)

    included = sum(
        row["status"] in {"included", "included_with_warning"}
        for row in report_rows
    )
    excluded = sum(row["status"] == "excluded" for row in report_rows)
    warnings = sum(row["status"] == "included_with_warning" for row in report_rows)

    print(f"Entrada: {input_path}")
    print(f"Documentos generados: {documents_output}")
    print(f"Reporte generado: {report_output}")
    print(f"Total registros: {len(report_rows)}")
    print(f"Incluidos: {included}")
    print(f"Incluidos con advertencia: {warnings}")
    print(f"Excluidos: {excluded}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
