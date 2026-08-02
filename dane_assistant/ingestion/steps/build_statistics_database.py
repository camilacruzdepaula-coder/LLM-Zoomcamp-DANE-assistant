"""Carga estadísticas normalizadas en SQLite para consultas del agente."""

import argparse
import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY,
    table_key TEXT NOT NULL,
    workbook TEXT NOT NULL,
    sheet TEXT NOT NULL,
    title TEXT NOT NULL,
    year TEXT,
    row_label TEXT NOT NULL,
    metric TEXT NOT NULL,
    unit TEXT NOT NULL,
    value REAL NOT NULL,
    source_url TEXT NOT NULL,
    source_page TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    source_column INTEGER NOT NULL
);
CREATE INDEX statistics_year_idx ON statistics(year);
CREATE INDEX statistics_table_idx ON statistics(table_key);
CREATE INDEX statistics_source_idx ON statistics(source_url);
CREATE VIRTUAL TABLE statistics_fts USING fts5(row_label, metric, title);
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    connection = sqlite3.connect(output_path)
    connection.executescript(SCHEMA)
    statistics_batch = []
    fts_batch = []
    inserted = 0

    with input_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            values = (
                row["table_key"], row["workbook"], row["sheet"], row["title"],
                row["year"], row["row_label"], row["metric"], row["unit"],
                row["value"], row["source_url"], row["source_page"],
                row["source_row"], row["source_column"],
            )
            statistics_batch.append(values)

            if len(statistics_batch) == 5000:
                connection.executemany(
                    """INSERT INTO statistics (
                    table_key, workbook, sheet, title, year, row_label, metric,
                    unit, value, source_url, source_page, source_row, source_column
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    statistics_batch,
                )
                start_id = inserted + 1
                fts_batch = [
                    (start_id + index, values[5], values[6], values[3])
                    for index, values in enumerate(statistics_batch)
                ]
                connection.executemany(
                    "INSERT INTO statistics_fts(rowid, row_label, metric, title) VALUES (?, ?, ?, ?)",
                    fts_batch,
                )
                inserted += len(statistics_batch)
                statistics_batch.clear()

    if statistics_batch:
        connection.executemany(
            """INSERT INTO statistics (
            table_key, workbook, sheet, title, year, row_label, metric,
            unit, value, source_url, source_page, source_row, source_column
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            statistics_batch,
        )
        start_id = inserted + 1
        connection.executemany(
            "INSERT INTO statistics_fts(rowid, row_label, metric, title) VALUES (?, ?, ?, ?)",
            [
                (start_id + index, values[5], values[6], values[3])
                for index, values in enumerate(statistics_batch)
            ],
        )
        inserted += len(statistics_batch)

    connection.commit()
    connection.close()
    print(f"Valores indexados: {inserted}")
    print(f"Base creada: {output_path}")


if __name__ == "__main__":
    main()
