"""Publica un artefacto local versionado en un dataset de Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(artifact_dir: Path, version: str) -> dict[str, Path]:
    archive = artifact_dir / f"dane-agent-data-{version}.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    manifest = artifact_dir / f"dane-agent-data-{version}.manifest.json"
    paths = {"archive": archive, "checksum": checksum, "manifest": manifest}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos locales: {missing}")
    return paths


def validate_artifact(paths: dict[str, Path], version: str) -> None:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != version:
        raise ValueError("La versión del manifest no coincide con --version.")
    archive_sha256 = sha256_file(paths["archive"])
    if archive_sha256 != manifest.get("archive", {}).get("sha256"):
        raise ValueError("El checksum del archivo local no coincide con su manifest.")
    checksum_value = paths["checksum"].read_text(encoding="utf-8").split()[0]
    if checksum_value != archive_sha256:
        raise ValueError("El archivo .sha256 no coincide con el artefacto local.")


def publish_artifact(
    artifact_dir: Path,
    version: str,
    repo_id: str,
    token: str,
) -> None:
    if not repo_id:
        raise ValueError("HF_DATASET_REPOSITORY es obligatorio para publicar.")
    if not token:
        raise ValueError("HF_TOKEN es obligatorio para publicar.")

    paths = artifact_paths(artifact_dir, version)
    validate_artifact(paths, version)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    prefix = f"releases/{version}"
    for name, path in paths.items():
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{prefix}/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Publish DANE agent data {version}: {name}",
        )
    print(f"Artefacto {version} publicado en {repo_id}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--version", default=os.getenv("DATA_VERSION", "v1"))
    parser.add_argument("--repo-id", default=os.getenv("HF_DATASET_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    publish_artifact(
        Path(args.artifact_dir),
        args.version,
        args.repo_id,
        args.token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
