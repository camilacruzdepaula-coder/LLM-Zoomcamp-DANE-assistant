#!/usr/bin/env python3
"""
Extrae contenido de PDF, Excel y páginas HTML para el proyecto DANE RAG.

Ejemplos:
    python extract_documents.py \
      --pdf "assets/2021/boletin.pdf" \
      --excel "assets/2021/anexo.xlsx"

    python extract_documents.py \
      --html "assets/2021/landing-id-2021.html"

    python extract_documents.py \
      --pdf "assets/2021/boletin.pdf" \
      --excel "assets/2021/anexo.xlsx" \
      --html "assets/2021/landing-id-2021.html"

Salidas:
    output/extracted_records.jsonl
    output/extraction_report.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_string(value: Any) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()

    return normalize_whitespace(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def infer_year_and_date(*values: Any) -> Tuple[Optional[int], Optional[str]]:
    joined = " ".join(normalize_whitespace(v) for v in values if v)

    date_match = re.search(
        r"\b((?:19|20)\d{2})[-_/\.]([01]\d)[-_/\.]([0-3]\d)\b",
        joined,
    )

    if date_match:
        year, month, day = date_match.groups()

        try:
            parsed = datetime(int(year), int(month), int(day))
            return int(year), parsed.date().isoformat()
        except ValueError:
            pass

    year_match = re.search(r"\b((?:19|20)\d{2})\b", joined)

    if year_match:
        return int(year_match.group(1)), None

    return None, None


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe el manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"El manifest está vacío: {manifest_path}")

    return rows


def first_nonempty(row: Dict[str, str], candidate_columns: List[str]) -> Optional[str]:
    for column in candidate_columns:
        value = row.get(column)

        if value and normalize_whitespace(value):
            return normalize_whitespace(value)

    return None


def resolve_local_path(
    raw_path: str,
    manifest_path: Path,
    assets_dir: Path,
) -> Optional[Path]:
    if not raw_path:
        return None

    candidate = Path(raw_path)

    candidates = [
        candidate,
        manifest_path.parent / candidate,
        manifest_path.parent.parent / candidate,
        assets_dir / candidate.name,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path.resolve()

    return None


def build_manifest_metadata(
    row: Optional[Dict[str, str]],
    local_path: Path,
) -> Dict[str, Any]:
    row = row or {}

    url = first_nonempty(
        row,
        [
            "url",
            "asset_url",
            "download_url",
            "file_url",
            "document_url",
            "original_url",
        ],
    )

    source_url = first_nonempty(
        row,
        [
            "source_url",
            "page_url",
            "landing_url",
            "parent_url",
            "origin_url",
            "referer_url",
        ],
    ) or url

    title = first_nonempty(
        row,
        [
            "title",
            "document_title",
            "name",
            "filename",
            "file_name",
            "label",
        ],
    ) or local_path.stem

    manifest_date = first_nonempty(
        row,
        [
            "date",
            "publication_date",
            "published_date",
            "fecha",
            "fecha_publicacion",
        ],
    )

    manifest_year = first_nonempty(
        row,
        [
            "year",
            "publication_year",
            "anio",
            "año",
            "inferred_year",
        ],
    )

    inferred_year, inferred_date = infer_year_and_date(
        manifest_year,
        manifest_date,
        title,
        url,
        source_url,
        local_path.name,
    )

    return {
        "url": url,
        "source_url": source_url,
        "title": title,
        "inferred_year": inferred_year,
        "inferred_date": inferred_date,
    }


def find_row_by_url(
    rows: List[Dict[str, str]],
    requested_url: str,
) -> Optional[Dict[str, str]]:
    requested_url = normalize_whitespace(requested_url)

    url_columns = [
        "url",
        "asset_url",
        "download_url",
        "file_url",
        "document_url",
        "original_url",
    ]

    for row in rows:
        for column in url_columns:
            value = normalize_whitespace(row.get(column, ""))

            if value == requested_url:
                return row

    return None


def find_file_path_from_row(
    row: Dict[str, str],
    manifest_path: Path,
    assets_dir: Path,
) -> Optional[Path]:
    local_path_columns = [
        "local_path",
        "path",
        "file_path",
        "asset_path",
        "download_path",
        "saved_path",
        "filepath",
    ]

    for column in local_path_columns:
        resolved = resolve_local_path(
            row.get(column, ""),
            manifest_path,
            assets_dir,
        )

        if resolved:
            return resolved

    return None


def choose_by_local_path(
    local_path: Optional[str],
    expected_extensions: Tuple[str, ...],
    manifest_path: Path,
    assets_dir: Path,
    manifest_rows: List[Dict[str, str]],
) -> Tuple[Optional[Path], Optional[Dict[str, str]]]:
    if not local_path:
        return None, None

    path = Path(local_path).expanduser().resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No existe el archivo indicado: {path}")

    if path.suffix.lower() not in expected_extensions:
        raise ValueError(
            f"Extensión no válida: {path.suffix}. "
            f"Se esperaba: {', '.join(expected_extensions)}"
        )

    matched_row = None

    for row in manifest_rows:
        candidate = find_file_path_from_row(row, manifest_path, assets_dir)

        if candidate and candidate.resolve() == path:
            matched_row = row
            break

    return path, matched_row


def choose_by_manifest_url(
    requested_url: Optional[str],
    expected_extensions: Tuple[str, ...],
    manifest_path: Path,
    assets_dir: Path,
    manifest_rows: List[Dict[str, str]],
) -> Tuple[Optional[Path], Optional[Dict[str, str]]]:
    if not requested_url:
        return None, None

    row = find_row_by_url(manifest_rows, requested_url)

    if row is None:
        raise ValueError(f"No se encontró esta URL en el manifest:\n{requested_url}")

    local_path = find_file_path_from_row(row, manifest_path, assets_dir)

    if local_path is None:
        raise FileNotFoundError(
            "La URL existe en el manifest, pero no se encontró el archivo local."
        )

    if local_path.suffix.lower() not in expected_extensions:
        raise ValueError(
            f"Extensión no válida: {local_path.suffix}. "
            f"Se esperaba: {', '.join(expected_extensions)}"
        )

    return local_path, row


def make_record(
    *,
    record_id: str,
    record_type: str,
    text: str,
    file_path: Path,
    file_sha256: str,
    metadata: Dict[str, Any],
    page_number: Optional[int] = None,
    sheet_name: Optional[str] = None,
    row_count: Optional[int] = None,
    column_names: Optional[List[str]] = None,
    table_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "text": normalize_whitespace(text),
        "provenance": {
            "url": metadata.get("url"),
            "source_url": metadata.get("source_url"),
            "title": metadata.get("title"),
            "local_path": str(file_path),
            "filename": file_path.name,
            "sha256": file_sha256,
            "inferred_year": metadata.get("inferred_year"),
            "inferred_date": metadata.get("inferred_date"),
            "page": page_number,
            "sheet": sheet_name,
        },
        "structure": {
            "row_count": row_count,
            "column_names": column_names,
            "table_rows": table_rows,
        },
        "extracted_at": utc_now(),
    }


def dataframe_to_text(
    dataframe: pd.DataFrame,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    dataframe = dataframe.copy()
    dataframe = dataframe.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if dataframe.empty:
        return "", [], []

    normalized_columns = [
        normalize_whitespace(column) or f"column_{index + 1}"
        for index, column in enumerate(dataframe.columns)
    ]

    dataframe.columns = normalized_columns

    table_rows: List[Dict[str, Any]] = []
    text_lines: List[str] = []

    for row_number, (_, row) in enumerate(dataframe.iterrows(), start=1):
        normalized_row = {
            column: json_safe(row[column])
            for column in normalized_columns
        }

        table_rows.append(normalized_row)

        row_text = " | ".join(
            f"{column}: {safe_string(normalized_row[column])}"
            for column in normalized_columns
            if safe_string(normalized_row[column])
        )

        if row_text:
            text_lines.append(f"Fila {row_number}: {row_text}")

    return "\n".join(text_lines), normalized_columns, table_rows


def extract_pdf(
    pdf_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    file_sha256 = sha256_file(pdf_path)
    records: List[Dict[str, Any]] = []
    pages_with_text = 0

    document = fitz.open(pdf_path)

    try:
        total_pages = len(document)

        for page_index in range(total_pages):
            page = document.load_page(page_index)
            text = normalize_whitespace(page.get_text("text"))

            if not text:
                continue

            pages_with_text += 1

            records.append(
                make_record(
                    record_id=f"{file_sha256}:page:{page_index + 1}",
                    record_type="pdf_page",
                    text=text,
                    file_path=pdf_path,
                    file_sha256=file_sha256,
                    metadata=metadata,
                    page_number=page_index + 1,
                )
            )
    finally:
        document.close()

    report = {
        "timestamp_utc": utc_now(),
        "file_type": "pdf",
        "local_path": str(pdf_path),
        "filename": pdf_path.name,
        "url": metadata.get("url"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "sha256": file_sha256,
        "status": "success",
        "records_written": len(records),
        "pages_total": total_pages,
        "pages_with_text": pages_with_text,
        "sheets_total": "",
        "sheets_extracted": "",
        "error_type": "",
        "error_message": "",
    }

    return records, report


def extract_excel(
    excel_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    suffix = excel_path.suffix.lower()

    if suffix == ".xls":
        engine = "xlrd"
    elif suffix in {".xlsx", ".xlsm"}:
        engine = "openpyxl"
    else:
        raise ValueError(f"Formato Excel no compatible: {suffix}")

    file_sha256 = sha256_file(excel_path)
    records: List[Dict[str, Any]] = []

    excel_file = pd.ExcelFile(excel_path, engine=engine)
    sheet_names = excel_file.sheet_names
    sheets_extracted = 0

    for sheet_name in sheet_names:
        dataframe = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            header=None,
            dtype=object,
        )

        dataframe.columns = [
            f"column_{index + 1}"
            for index in range(dataframe.shape[1])
        ]

        text, column_names, table_rows = dataframe_to_text(dataframe)

        if not table_rows:
            continue

        sheets_extracted += 1

        records.append(
            make_record(
                record_id=f"{file_sha256}:sheet:{sheet_name}",
                record_type="excel_sheet",
                text=f"Hoja: {sheet_name}\n{text}",
                file_path=excel_path,
                file_sha256=file_sha256,
                metadata=metadata,
                sheet_name=sheet_name,
                row_count=len(table_rows),
                column_names=column_names,
                table_rows=table_rows,
            )
        )

    report = {
        "timestamp_utc": utc_now(),
        "file_type": "excel",
        "local_path": str(excel_path),
        "filename": excel_path.name,
        "url": metadata.get("url"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "sha256": file_sha256,
        "status": "success",
        "records_written": len(records),
        "pages_total": "",
        "pages_with_text": "",
        "sheets_total": len(sheet_names),
        "sheets_extracted": sheets_extracted,
        "error_type": "",
        "error_message": "",
    }

    return records, report


def extract_dane_html_content(html_path: Path) -> Tuple[str, str]:
    """
    Extrae título y contenido editorial de una página DANE.

    Se limita a div[itemprop="articleBody"] para no indexar navegación,
    footer, botones de compartir, scripts ni otros componentes de interfaz.
    """
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    headline = soup.select_one("h2[itemprop='headline']")
    article_body = soup.select_one("div[itemprop='articleBody']")

    if article_body is None:
        raise ValueError(
            "No se encontró div[itemprop='articleBody'] en el archivo HTML."
        )

    for element in article_body.select(
        "script, style, nav, footer, .addtoany_container"
    ):
        element.decompose()

    title = ""

    if headline is not None:
        title = normalize_whitespace(headline.get_text(" ", strip=True))

    content = normalize_whitespace(article_body.get_text("\n", strip=True))

    if not content:
        raise ValueError(
            "div[itemprop='articleBody'] no contiene texto extraíble."
        )

    return title, content


def extract_html(
    html_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    file_sha256 = sha256_file(html_path)
    title, content = extract_dane_html_content(html_path)

    if title:
        metadata = {**metadata, "title": title}

    record = make_record(
        record_id=f"{file_sha256}:html:article",
        record_type="html_article",
        text=content,
        file_path=html_path,
        file_sha256=file_sha256,
        metadata=metadata,
    )

    report = {
        "timestamp_utc": utc_now(),
        "file_type": "html",
        "local_path": str(html_path),
        "filename": html_path.name,
        "url": metadata.get("url"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "sha256": file_sha256,
        "status": "success",
        "records_written": 1,
        "pages_total": "",
        "pages_with_text": "",
        "sheets_total": "",
        "sheets_extracted": "",
        "error_type": "",
        "error_message": "",
    }

    return [record], report


def build_error_report(
    file_type: str,
    file_path: Optional[Path],
    metadata: Dict[str, Any],
    error: Exception,
) -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "file_type": file_type,
        "local_path": str(file_path) if file_path else "",
        "filename": file_path.name if file_path else "",
        "url": metadata.get("url"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "sha256": "",
        "status": "error",
        "records_written": 0,
        "pages_total": "",
        "pages_with_text": "",
        "sheets_total": "",
        "sheets_extracted": "",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_safe))
            handle.write("\n")


def write_report(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fields = [
        "timestamp_utc",
        "file_type",
        "local_path",
        "filename",
        "url",
        "source_url",
        "title",
        "sha256",
        "status",
        "records_written",
        "pages_total",
        "pages_with_text",
        "sheets_total",
        "sheets_extracted",
        "error_type",
        "error_message",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae contenido de PDF, Excel y HTML para DANE RAG."
    )

    parser.add_argument(
        "--manifest",
        default="manifests/manifest.csv",
        help="Ruta a manifest.csv.",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Directorio con archivos originales.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directorio para archivos procesados.",
    )

    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument("--pdf", help="Ruta local del PDF.")
    pdf_group.add_argument("--pdf-url", help="URL del PDF en manifest.csv.")

    excel_group = parser.add_mutually_exclusive_group()
    excel_group.add_argument("--excel", help="Ruta local del Excel.")
    excel_group.add_argument("--excel-url", help="URL del Excel en manifest.csv.")

    html_group = parser.add_mutually_exclusive_group()
    html_group.add_argument("--html", help="Ruta local del archivo HTML.")
    html_group.add_argument("--html-url", help="URL del HTML en manifest.csv.")

    args = parser.parse_args()

    if not any(
        [
            args.pdf,
            args.pdf_url,
            args.excel,
            args.excel_url,
            args.html,
            args.html_url,
        ]
    ):
        parser.error(
            "Debes indicar al menos un documento: PDF, Excel o HTML."
        )

    return args


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    records_output = output_dir / "extracted_records.jsonl"
    report_output = output_dir / "extraction_report.csv"

    all_records: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    try:
        manifest_rows = read_manifest(manifest_path)
    except Exception as error:
        print(f"ERROR leyendo el manifest: {error}", file=sys.stderr)
        return 2

    if args.pdf or args.pdf_url:
        pdf_path: Optional[Path] = None
        pdf_metadata: Dict[str, Any] = {}

        try:
            if args.pdf:
                pdf_path, pdf_row = choose_by_local_path(
                    args.pdf,
                    (".pdf",),
                    manifest_path,
                    assets_dir,
                    manifest_rows,
                )
            else:
                pdf_path, pdf_row = choose_by_manifest_url(
                    args.pdf_url,
                    (".pdf",),
                    manifest_path,
                    assets_dir,
                    manifest_rows,
                )

            assert pdf_path is not None
            pdf_metadata = build_manifest_metadata(pdf_row, pdf_path)

            pdf_records, pdf_report = extract_pdf(pdf_path, pdf_metadata)
            all_records.extend(pdf_records)
            report_rows.append(pdf_report)

            print(
                f"PDF procesado: {pdf_path.name} "
                f"({len(pdf_records)} páginas con texto)."
            )

        except Exception as error:
            traceback.print_exc()
            report_rows.append(
                build_error_report("pdf", pdf_path, pdf_metadata, error)
            )
            print(f"ERROR procesando PDF: {error}", file=sys.stderr)

    if args.excel or args.excel_url:
        excel_path: Optional[Path] = None
        excel_metadata: Dict[str, Any] = {}

        try:
            if args.excel:
                excel_path, excel_row = choose_by_local_path(
                    args.excel,
                    (".xlsx", ".xls", ".xlsm"),
                    manifest_path,
                    assets_dir,
                    manifest_rows,
                )
            else:
                excel_path, excel_row = choose_by_manifest_url(
                    args.excel_url,
                    (".xlsx", ".xls", ".xlsm"),
                    manifest_path,
                    assets_dir,
                    manifest_rows,
                )

            assert excel_path is not None
            excel_metadata = build_manifest_metadata(excel_row, excel_path)

            excel_records, excel_report = extract_excel(excel_path, excel_metadata)
            all_records.extend(excel_records)
            report_rows.append(excel_report)

            print(
                f"Excel procesado: {excel_path.name} "
                f"({len(excel_records)} hojas no vacías)."
            )

        except Exception as error:
            traceback.print_exc()
            report_rows.append(
                build_error_report("excel", excel_path, excel_metadata, error)
            )
            print(f"ERROR procesando Excel: {error}", file=sys.stderr)

    if args.html or args.html_url:
        html_path: Optional[Path] = None
        html_metadata: Dict[str, Any] = {}

        try:
            if args.html:
                html_path, html_row = choose_by_local_path(
                    args.html,
                    (".html", ".htm"),
                    manifest_path,
                    assets_dir,
                    manifest_rows,
                )
            else:
                html_path, html_row = choose_by_manifest_url(
                    args.html_url,
                    (".html", ".htm"),
                    manifest_path,
                    assets_dir,
                    manifest_rows,
                )

            assert html_path is not None
            html_metadata = build_manifest_metadata(html_row, html_path)

            html_records, html_report = extract_html(html_path, html_metadata)
            all_records.extend(html_records)
            report_rows.append(html_report)

            print(
                f"HTML procesado: {html_path.name} "
                f"({len(html_records)} artículo extraído)."
            )

        except Exception as error:
            traceback.print_exc()
            report_rows.append(
                build_error_report("html", html_path, html_metadata, error)
            )
            print(f"ERROR procesando HTML: {error}", file=sys.stderr)

    write_jsonl(all_records, records_output)
    write_report(report_rows, report_output)

    print(f"\nJSONL generado: {records_output}")
    print(f"Reporte generado: {report_output}")
    print(f"Registros escritos: {len(all_records)}")

    if any(row["status"] == "error" for row in report_rows):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())