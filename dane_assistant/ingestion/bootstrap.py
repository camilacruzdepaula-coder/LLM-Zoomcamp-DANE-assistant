"""Descarga y valida un artefacto público de datos para el contenedor de la app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen


HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"
RUNTIME_ROOTS = ("statistics", "rag")
VERSION_MARKER = ".dane_data_version"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "dane-agent-data-init/1.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def huggingface_url(repo_id: str, revision: str, path: str) -> str:
    quoted_path = "/".join(quote(part) for part in path.split("/"))
    return HF_RESOLVE_URL.format(repo_id=repo_id, revision=revision, path=quoted_path)


def safe_extract(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("El archivo de datos está vacío.")
        root_names = {PurePosixPath(member.name).parts[0] for member in members}
        if len(root_names) != 1:
            raise ValueError("El artefacto debe contener una única carpeta raíz.")
        for member in members:
            member_path = destination / member.name
            if not member_path.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Ruta insegura en el artefacto: {member.name}")
        archive.extractall(destination, filter="data")
    return destination / root_names.pop()


def validate_extracted_data(extracted_root: Path, metadata: dict) -> None:
    for file_info in metadata["files"]:
        path = extracted_root / file_info["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Falta el archivo esperado: {path}")
        if sha256_file(path) != file_info["sha256"]:
            raise ValueError(f"Checksum inválido para: {file_info['path']}")


def install_artifact(data_dir: Path, extracted_root: Path, version: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for root_name in RUNTIME_ROOTS:
        source = extracted_root / root_name
        if not source.exists():
            raise FileNotFoundError(f"El artefacto no contiene: {root_name}")
        destination = data_dir / root_name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), destination)
    (data_dir / VERSION_MARKER).write_text(version, encoding="utf-8")


def data_is_current(data_dir: Path, version: str) -> bool:
    marker = data_dir / VERSION_MARKER
    return (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == version
        and all((data_dir / root_name).is_dir() for root_name in RUNTIME_ROOTS)
    )


def bootstrap_data(data_dir: Path, repo_id: str, revision: str, version: str) -> None:
    if not repo_id:
        raise ValueError("HF_DATASET_REPOSITORY debe indicar el dataset público.")
    if data_is_current(data_dir, version):
        print(f"Los datos {version} ya están disponibles en {data_dir}.")
        return

    release_prefix = f"releases/{version}"
    manifest_name = f"dane-agent-data-{version}.manifest.json"
    manifest_url = huggingface_url(repo_id, revision, f"{release_prefix}/{manifest_name}")
    with urlopen(Request(manifest_url), timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    if metadata.get("artifact_version") != version:
        raise ValueError("El manifest descargado no coincide con la versión solicitada.")

    archive_name = metadata["archive"]["path"]
    archive_url = huggingface_url(repo_id, revision, f"{release_prefix}/{archive_name}")
    with tempfile.TemporaryDirectory(dir=data_dir.parent) as temporary_directory:
        temporary_dir = Path(temporary_directory)
        archive_path = temporary_dir / archive_name
        print(f"Descargando datos {version} desde Hugging Face...")
        download(archive_url, archive_path)
        if sha256_file(archive_path) != metadata["archive"]["sha256"]:
            raise ValueError("Checksum inválido para el archivo descargado.")
        extracted_root = safe_extract(archive_path, temporary_dir / "extracted")
        validate_extracted_data(extracted_root, metadata)
        install_artifact(data_dir, extracted_root, version)
    print(f"Datos {version} instalados en {data_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.getenv("DANE_DATA_DIR", "/data"))
    parser.add_argument("--repo-id", default=os.getenv("HF_DATASET_REPOSITORY", ""))
    parser.add_argument("--revision", default=os.getenv("HF_DATASET_REVISION", "main"))
    parser.add_argument("--version", default=os.getenv("DATA_VERSION", "v1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bootstrap_data(Path(args.data_dir), args.repo_id, args.revision, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
