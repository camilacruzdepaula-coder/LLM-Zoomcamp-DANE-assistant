"""Flujo Prefect para reconstruir y empaquetar el conocimiento del agente DANE.

Este módulo no publica datos. Construye un artefacto local versionado que puede
subirse desde otra máquina a Hugging Face u otro almacenamiento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow, get_run_logger, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATHS = (
    Path("statistics/statistics.db"),
    Path("rag/chunks.jsonl"),
    Path("rag/embeddings.npy"),
    Path("rag/bm25_index.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_statistics_database(database_path: Path) -> dict[str, int]:
    if not database_path.is_file() or database_path.stat().st_size == 0:
        raise FileNotFoundError(f"No existe una base estadística válida: {database_path}")

    with sqlite3.connect(database_path) as connection:
        statistics_count = connection.execute("SELECT COUNT(*) FROM statistics").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM statistics_fts").fetchone()[0]

    if statistics_count == 0:
        raise ValueError("La base estadística no contiene filas.")
    if statistics_count != fts_count:
        raise ValueError(
            "El índice FTS no coincide con la tabla statistics: "
            f"{statistics_count} != {fts_count}."
        )
    return {"statistics_rows": statistics_count, "fts_rows": fts_count}


def validate_rag_index(index_dir: Path) -> dict[str, int]:
    chunks_path = index_dir / "chunks.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    bm25_path = index_dir / "bm25_index.json"
    missing = [path for path in (chunks_path, embeddings_path, bm25_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos del índice RAG: {missing}")

    chunk_count = count_jsonl_records(chunks_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    embedding_count = int(embeddings.shape[0])
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    bm25_count = int(bm25["document_count"])

    if not chunk_count:
        raise ValueError("El índice RAG no contiene chunks.")
    if chunk_count != embedding_count or chunk_count != bm25_count:
        raise ValueError(
            "El índice RAG es inconsistente: "
            f"chunks={chunk_count}, embeddings={embedding_count}, bm25={bm25_count}."
        )
    return {
        "chunks": chunk_count,
        "embeddings": embedding_count,
        "bm25_documents": bm25_count,
    }


def build_release_metadata(build_root: Path, version: str) -> dict[str, Any]:
    files = []
    for relative_path in RUNTIME_PATHS:
        path = build_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"No se puede empaquetar el archivo faltante: {path}")
        files.append(
            {
                "path": relative_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "artifact_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def create_runtime_artifact(
    build_root: Path,
    artifact_dir: Path,
    version: str,
) -> dict[str, Path]:
    metadata = build_release_metadata(build_root, version)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"dane-agent-data-{version}.tar.gz"
    checksum_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
    manifest_path = artifact_dir / f"dane-agent-data-{version}.manifest.json"

    with tempfile.TemporaryDirectory(dir=artifact_dir) as temporary_directory:
        staging_root = Path(temporary_directory) / f"dane-agent-data-{version}"
        for relative_path in RUNTIME_PATHS:
            source = build_root / relative_path
            destination = staging_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (staging_root / "release_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with tarfile.open(artifact_path, "w:gz") as archive:
            archive.add(staging_root, arcname=staging_root.name)

    artifact_sha256 = sha256_file(artifact_path)
    checksum_path.write_text(
        f"{artifact_sha256}  {artifact_path.name}\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                **metadata,
                "archive": {
                    "path": artifact_path.name,
                    "bytes": artifact_path.stat().st_size,
                    "sha256": artifact_sha256,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "artifact": artifact_path,
        "checksum": checksum_path,
        "manifest": manifest_path,
    }


@task(name="Prepare ingestion workspace")
def prepare_workspace(build_root: Path, overwrite: bool) -> None:
    if build_root.exists() and any(build_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"El directorio de salida ya contiene archivos: {build_root}. "
                "Usa --overwrite para reemplazarlo."
            )
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)


@task(name="Run ingestion script")
def run_python_module(step_name: str, module_name: str, arguments: list[str]) -> None:
    logger = get_run_logger()
    command = [sys.executable, "-m", module_name, *arguments]
    logger.info("%s: %s", step_name, " ".join(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        logger.info("%s output:\n%s", step_name, result.stdout.rstrip())
    if result.returncode:
        raise RuntimeError(
            f"{step_name} falló con código {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if result.stderr:
        logger.warning("%s warnings:\n%s", step_name, result.stderr.rstrip())


@task(name="Validate runtime data")
def validate_runtime_data(runtime_root: Path) -> dict[str, int]:
    statistics = validate_statistics_database(
        runtime_root / "statistics/statistics.db"
    )
    rag = validate_rag_index(runtime_root / "rag")
    return {**statistics, **rag}


@task(name="Package runtime artifact")
def package_runtime_artifact(
    runtime_root: Path, artifact_dir: Path, version: str
) -> dict[str, str]:
    paths = create_runtime_artifact(runtime_root, artifact_dir, version)
    return {name: str(path) for name, path in paths.items()}


@flow(name="dane-technology-ingestion")
def dane_technology_ingestion(
    build_root: Path,
    artifact_dir: Path,
    version: str,
    overwrite: bool = False,
    scrape_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Descarga fuentes DANE, reconstruye los índices y genera un artefacto local."""
    build_root = build_root.resolve()
    artifact_dir = artifact_dir.resolve()
    source_root = build_root / "sources"
    technology_root = source_root / "technology"
    normalized_dir = technology_root / "normalized"
    documents_root = source_root / "documents"
    processed_dir = documents_root / "processed"
    runtime_root = build_root / "data"
    statistics_dir = runtime_root / "statistics"
    rag_dir = runtime_root / "rag"
    manifest_path = technology_root / "manifest.jsonl"

    prepare_workspace(build_root, overwrite)
    run_python_module.with_options(retries=2, retry_delay_seconds=30)(
        "Scrape fuentes oficiales DANE",
        "dane_assistant.ingestion.steps.scrape_dane_technology",
        ["--output", str(technology_root), "--delay", str(scrape_delay_seconds)],
    )

    normalize_future = run_python_module.submit(
        "Normaliza anexos tabulares",
        "dane_assistant.ingestion.steps.normalize_tabular_sources",
        ["--manifest", str(manifest_path), "--output-dir", str(normalized_dir)],
    )
    extract_future = run_python_module.submit(
        "Extrae documentos para RAG",
        "dane_assistant.ingestion.steps.ingest_dane_technology_documents",
        [
            "--manifest", str(manifest_path),
            "--records-output", str(processed_dir / "extracted_records.jsonl"),
            "--report-output", str(processed_dir / "extraction_report.csv"),
        ],
    )

    normalize_future.result()
    statistics_future = run_python_module.submit(
        "Construye base estadística SQLite",
        "dane_assistant.ingestion.steps.build_statistics_database",
        [
            "--input", str(normalized_dir / "statistics.jsonl"),
            "--output", str(statistics_dir / "statistics.db"),
        ],
    )

    extract_future.result()
    run_python_module(
        "Normaliza documentos RAG",
        "dane_assistant.ingestion.steps.build_documents",
        [
            "--input", str(processed_dir / "extracted_records.jsonl"),
            "--output-dir", str(processed_dir),
            "--topic", "Tecnología e Innovación",
            "--subtopic", "Fuentes oficiales DANE de tecnología e innovación",
        ],
    )
    run_python_module(
        "Crea chunks RAG",
        "dane_assistant.ingestion.steps.build_chunks",
        ["--input", str(processed_dir / "documents.jsonl"), "--output", str(processed_dir / "chunks.jsonl")],
    )
    run_python_module(
        "Genera embeddings e índice BM25",
        "dane_assistant.ingestion.steps.embed_chunks",
        ["--input", str(processed_dir / "chunks.jsonl"), "--output-dir", str(rag_dir)],
    )
    statistics_future.result()

    validation = validate_runtime_data(runtime_root)
    artifact = package_runtime_artifact(runtime_root, artifact_dir, version)
    return {"validation": validation, "artifact": artifact}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", default="build/dane-ingestion")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--version", required=True, help="Ejemplo: v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scrape-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra las rutas de salida sin descargar ni procesar datos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("No se ejecutó la ingestión.")
        print(f"Build local: {Path(args.build_root).resolve()}")
        print(f"Artefactos locales: {Path(args.artifact_dir).resolve()}")
        return 0

    result = dane_technology_ingestion(
        build_root=Path(args.build_root),
        artifact_dir=Path(args.artifact_dir),
        version=args.version,
        overwrite=args.overwrite,
        scrape_delay_seconds=args.scrape_delay_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
