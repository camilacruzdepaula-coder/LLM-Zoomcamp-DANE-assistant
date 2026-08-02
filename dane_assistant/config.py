"""Rutas configurables para datos de runtime locales o montados por Docker."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(os.getenv("DANE_DATA_DIR", "data"))
STATISTICS_DATABASE_PATH = Path(
    os.getenv(
        "DANE_STATISTICS_DB",
        DATA_ROOT / "statistics/statistics.db",
    )
)
RAG_INDEX_DIR = Path(
    os.getenv("DANE_RAG_INDEX_DIR", DATA_ROOT / "rag")
)
