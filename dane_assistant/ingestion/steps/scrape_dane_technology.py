"""Descarga HTML y documentos tabulares/PDF de páginas DANE seleccionadas."""

import argparse
import csv
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


SEED_URLS = [
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/tecnologia-e-innovacion/encuesta-de-desarrollo-e-innovacion-tecnologica-edit",
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/tecnologia-e-innovacion/tecnologias-de-la-informacion-y-las-comunicaciones-tic/indicadores-basicos-de-tic-en-hogares",
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/tecnologia-e-innovacion/tecnologias-de-la-informacion-y-las-comunicaciones-tic/indicadores-basicos-de-tic-en-empresas",
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/tecnologia-e-innovacion/tecnologias-de-la-informacion-y-las-comunicaciones-tic/encuesta-de-tecnologias-de-la-informacion-y-las-comunicaciones-en-hogares-entic-hogares",
    "https://www.dane.gov.co/index.php/estadisticas-por-tema/tecnologia-e-innovacion/tecnologias-de-la-informacion-y-las-comunicaciones-tic/encuesta-de-tecnologias-de-la-informacion-y-las-comunicaciones-en-empresas-entic-empresas",
]

DOCUMENT_EXTENSIONS = {".pdf", ".xls", ".xlsx", ".csv"}
ALLOWED_HOSTS = {"www.dane.gov.co", "dane.gov.co"}


@dataclass
class ManifestRecord:
    resource_type: str
    url: str
    source_page: str
    link_text: str
    context: str
    local_path: str
    content_type: str
    sha256: str
    status: str
    error: str = ""


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def safe_filename(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or fallback


def asset_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def page_slug(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return safe_filename(Path(path).name, "pagina")


class DaneTechnologyScraper:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = Path(args.output)
        self.html_dir = self.output / "html"
        self.assets_dir = self.output / "assets"
        self.manifest: list[ManifestRecord] = []
        self.seen_assets: set[str] = set()
        self.robots = self._load_robots()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": args.user_agent})

    def _load_robots(self) -> RobotFileParser:
        parser = RobotFileParser()
        robots_url = "https://www.dane.gov.co/robots.txt"
        response = requests.get(robots_url, timeout=self.args.timeout)
        response.raise_for_status()
        parser.parse(response.text.splitlines())
        return parser

    def _check_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"URL fuera de alcance: {url}")
        if not self.args.ignore_robots and not self.robots.can_fetch(self.args.user_agent, url):
            raise PermissionError(f"robots.txt no permite acceder a: {url}")

    def _request(self, url: str, stream: bool = False) -> requests.Response:
        self._check_url(url)
        response = self.session.get(url, timeout=self.args.timeout, stream=stream)
        response.raise_for_status()
        final_url = response.url
        self._check_url(final_url)
        return response

    def _record(self, **kwargs: str) -> None:
        self.manifest.append(ManifestRecord(**kwargs))

    def _save_html(self, url: str, content: bytes) -> Path:
        self.html_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        path = self.html_dir / f"{page_slug(url)}_{digest}.html"
        path.write_bytes(content)
        return path

    def _extract_assets(self, html: bytes, page_url: str) -> Iterable[tuple[str, str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            url = urljoin(page_url, anchor["href"].strip())
            if asset_extension(url) not in DOCUMENT_EXTENSIONS:
                continue

            link_text = normalize_space(anchor.get_text(" ", strip=True))
            parent = anchor.parent.get_text(" ", strip=True) if anchor.parent else ""
            context = normalize_space(parent)
            yield url, link_text, context

    def _asset_path(self, url: str, source_page: str) -> Path:
        extension = asset_extension(url)
        name = safe_filename(Path(urlparse(url).path).stem, "documento")
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        filename = f"{name}_{digest}{extension}"
        path = self.assets_dir / page_slug(source_page) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _download_asset(self, url: str, source_page: str, link_text: str, context: str) -> None:
        if url in self.seen_assets:
            return
        self.seen_assets.add(url)

        resource_type = asset_extension(url).lstrip(".")
        try:
            if self.args.dry_run:
                self._record(
                    resource_type=resource_type,
                    url=url,
                    source_page=source_page,
                    link_text=link_text,
                    context=context,
                    local_path="",
                    content_type="",
                    sha256="",
                    status="dry_run",
                )
                return

            response = self._request(url, stream=True)
            destination = self._asset_path(response.url, source_page)
            digest = hashlib.sha256()
            with destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        file.write(chunk)
                        digest.update(chunk)

            self._record(
                resource_type=resource_type,
                url=response.url,
                source_page=source_page,
                link_text=link_text,
                context=context,
                local_path=str(destination),
                content_type=response.headers.get("Content-Type", ""),
                sha256=digest.hexdigest(),
                status="downloaded",
            )
        except (requests.RequestException, PermissionError, ValueError) as error:
            logging.warning("No se descargó %s: %s", url, error)
            self._record(
                resource_type=resource_type,
                url=url,
                source_page=source_page,
                link_text=link_text,
                context=context,
                local_path="",
                content_type="",
                sha256="",
                status="error",
                error=str(error),
            )

    def _scrape_page(self, url: str) -> None:
        try:
            response = self._request(url)
            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                raise ValueError(f"Se esperaba HTML y se recibió {content_type}")

            html = response.content
            local_path = ""
            if not self.args.dry_run:
                local_path = str(self._save_html(response.url, html))

            self._record(
                resource_type="html",
                url=response.url,
                source_page=response.url,
                link_text="",
                context="",
                local_path=local_path,
                content_type=content_type,
                sha256=hashlib.sha256(html).hexdigest(),
                status="downloaded" if not self.args.dry_run else "dry_run",
            )

            for asset_url, link_text, context in self._extract_assets(html, response.url):
                self._download_asset(asset_url, response.url, link_text, context)
                time.sleep(self.args.delay)
        except (requests.RequestException, PermissionError, ValueError) as error:
            logging.warning("No se procesó %s: %s", url, error)
            self._record(
                resource_type="html",
                url=url,
                source_page=url,
                link_text="",
                context="",
                local_path="",
                content_type="",
                sha256="",
                status="error",
                error=str(error),
            )

    def _write_manifest(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.output / "manifest.jsonl"
        csv_path = self.output / "manifest.csv"

        with jsonl_path.open("w", encoding="utf-8") as file:
            for record in self.manifest:
                file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=ManifestRecord.__dataclass_fields__)
            writer.writeheader()
            writer.writerows(asdict(record) for record in self.manifest)

    def run(self) -> None:
        for position, url in enumerate(self.args.urls, start=1):
            logging.info("Procesando página %s/%s: %s", position, len(self.args.urls), url)
            self._scrape_page(url)
            time.sleep(self.args.delay)
        self._write_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        default=None,
        help="Página DANE semilla. Se puede repetir; por defecto usa las cinco páginas configuradas.",
    )
    parser.add_argument("--output", default="build/dane-ingestion/sources/technology")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--user-agent",
        default="DaneTechnologyAcademicScraper/1.0 (llm-zoomcamp project)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-robots", action="store_true")
    args = parser.parse_args()
    args.urls = args.urls or SEED_URLS
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scraper = DaneTechnologyScraper(args)
    scraper.run()
    downloaded = sum(record.status == "downloaded" for record in scraper.manifest)
    errors = sum(record.status == "error" for record in scraper.manifest)
    print(f"Recursos descargados: {downloaded}; errores: {errors}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
