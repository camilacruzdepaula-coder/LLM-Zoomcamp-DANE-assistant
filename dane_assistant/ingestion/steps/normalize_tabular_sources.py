"""Normaliza anexos Excel del DANE a valores numéricos en formato largo."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def clean(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def numeric_value(value):
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    return None


def find_title(frame):
    for value in frame.to_numpy().ravel():
        text = clean(value)
        if text.lower().startswith("cuadro"):
            return text
    return ""


def find_data_start(frame):
    for index, row in frame.iterrows():
        values = list(row)
        numeric_count = sum(numeric_value(value) is not None for value in values)
        text_count = sum(bool(clean(value)) and numeric_value(value) is None for value in values)
        if numeric_count >= 2 and text_count >= 1:
            return index
    return None


def header_paths(frame, data_start):
    paths = [[] for _ in frame.columns]
    for _, row in frame.iloc[:data_start].iterrows():
        carried = ""
        for column, value in enumerate(row):
            text = clean(value)
            if text:
                carried = text
            elif carried:
                text = carried

            lower = text.lower()
            if not text or lower.startswith("cuadro") or len(text) > 180:
                continue
            if text not in paths[column]:
                paths[column].append(text)
    return [" | ".join(parts) for parts in paths]


def infer_unit(title, metric):
    text = f"{title} {metric}".lower()
    if "miles de pesos" in text:
        return "miles de pesos"
    if "porcentaje" in text or "proporción" in text:
        return "porcentaje"
    if "unidades" in text or "número" in text or "numero" in text:
        return "unidades"
    return "sin especificar"


def infer_year(*values):
    years = []
    for value in values:
        years.extend(int(year) for year in YEAR_RE.findall(str(value or "")))
    return str(max(years)) if years else ""


def normalize_sheet(frame, source, workbook_name, sheet_name):
    title = find_title(frame)
    data_start = find_data_start(frame)
    if data_start is None:
        return [], {
            "workbook": workbook_name,
            "sheet": sheet_name,
            "title": title,
            "status": "unsupported_no_data_start",
        }

    paths = header_paths(frame, data_start)
    rows = []
    table_key = f"{Path(workbook_name).stem}:{sheet_name}"
    year = infer_year(title, workbook_name, source["url"])

    for frame_row, row in frame.iloc[data_start:].iterrows():
        values = list(row)
        numeric_columns = [
            column
            for column, value in enumerate(values)
            if numeric_value(value) is not None
        ]
        if not numeric_columns:
            continue

        first_numeric = min(numeric_columns)
        row_label = " | ".join(
            clean(value) for value in values[:first_numeric] if clean(value)
        )
        if not row_label:
            continue

        for column in numeric_columns:
            metric = paths[column] or f"columna_{column + 1}"
            value = numeric_value(values[column])
            rows.append(
                {
                    "table_key": table_key,
                    "workbook": workbook_name,
                    "sheet": sheet_name,
                    "title": title,
                    "year": year,
                    "row_label": row_label,
                    "metric": metric,
                    "unit": infer_unit(title, metric),
                    "value": value,
                    "source_url": source["url"],
                    "source_page": source["source_page"],
                    "source_row": int(frame_row) + 1,
                    "source_column": int(column) + 1,
                }
            )

    return rows, {
        "table_key": table_key,
        "workbook": workbook_name,
        "sheet": sheet_name,
        "title": title,
        "year": year,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "values_normalized": len(rows),
        "status": "normalized_generic_v1",
    }


def read_manifest(path):
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    normalized = []
    for record in records:
        local_path = Path(record["local_path"])
        if not local_path.is_absolute():
            local_path = path.parent.parent / local_path
        normalized.append(
            {
                **record,
                "local_path": str(local_path),
                "source_page": record.get("source_page") or record.get("source_url", ""),
            }
        )
    return normalized


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    value_rows = []
    catalog_rows = []
    workbooks = []
    for manifest_path in args.manifest:
        workbooks.extend(
            record
            for record in read_manifest(Path(manifest_path))
            if record.get("status") == "downloaded"
            and record.get("resource_type") in {"xls", "xlsx"}
        )

    for position, source in enumerate(workbooks, start=1):
        path = Path(source["local_path"])
        print(f"[{position}/{len(workbooks)}] {path.name}", flush=True)
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
            rows, catalog = normalize_sheet(frame, source, path.name, sheet_name)
            value_rows.extend(rows)
            catalog_rows.append(catalog)

    write_jsonl(value_rows, output_dir / "statistics.jsonl")
    write_jsonl(catalog_rows, output_dir / "table_catalog.jsonl")
    print(f"Valores normalizados: {len(value_rows)}")
    print(f"Tablas catalogadas: {len(catalog_rows)}")


if __name__ == "__main__":
    main()
