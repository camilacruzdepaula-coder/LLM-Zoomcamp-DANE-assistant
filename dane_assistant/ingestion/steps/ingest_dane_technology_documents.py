"""Ingiere PDF y HTML descargados desde el manifiesto DANE al corpus extraído."""

import argparse
import json
import re
from pathlib import Path

from dane_assistant.ingestion.steps.extract_documents import (
    extract_html,
    extract_pdf,
    write_jsonl,
    write_report,
)


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def infer_year(*values):
    years = []
    for value in values:
        years.extend(int(year) for year in YEAR_RE.findall(str(value or "")))
    return str(max(years)) if years else ""


def metadata_from_manifest(record):
    path = Path(record["local_path"])
    return {
        "url": record["url"],
        "source_url": record["source_page"],
        "title": record["link_text"] or path.name,
        "inferred_year": infer_year(record["url"], record["link_text"], record["context"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    existing_records = read_jsonl(Path(args.records_output))
    known_ids = {record.get("record_id") for record in existing_records}
    new_records = []
    reports = []

    for record in manifest:
        if record.get("status") != "downloaded":
            continue
        if record.get("resource_type") not in {"pdf", "html"}:
            continue

        path = Path(record["local_path"])
        metadata = metadata_from_manifest(record)
        try:
            if record["resource_type"] == "pdf":
                extracted, report = extract_pdf(path, metadata)
            else:
                extracted, report = extract_html(path, metadata)
        except Exception as error:
            reports.append(
                {
                    "file_type": record["resource_type"],
                    "local_path": str(path),
                    "filename": path.name,
                    "url": record["url"],
                    "source_url": record["source_page"],
                    "title": metadata["title"],
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue

        reports.append(report)
        for extracted_record in extracted:
            if extracted_record["record_id"] not in known_ids:
                new_records.append(extracted_record)
                known_ids.add(extracted_record["record_id"])

    records_path = Path(args.records_output)
    report_path = Path(args.report_output)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(existing_records + new_records, records_path)
    write_report(reports, report_path)
    print(f"Registros existentes: {len(existing_records)}")
    print(f"Registros nuevos: {len(new_records)}")
    print(f"Errores de extracción: {sum(report.get('status') == 'error' for report in reports)}")


if __name__ == "__main__":
    main()
