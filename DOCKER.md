# Docker deployment

Docker Compose runs the complete local application: it downloads the public data
artifact, starts Streamlit, and persists monitoring data in a named volume.

## 1. Configure the environment

Create `.env` in the repository root, next to `docker-compose.yml`.

Set `OPENAI_API_KEY`, `HF_DATASET_REPOSITORY`, and the intended `DATA_VERSION`.
The Hugging Face repository must be a public Dataset repository, for example
`your-user/dane-technology-rag-data`.

`HF_TOKEN` is not needed to run the application. It is used only by the explicit
publishing command below and must never be committed.

Before building an image for distribution, generate the dependency lockfile:

```bash
uv lock
```

## 2. Build the data artifact on the source computer

Run this only on the computer that has the source data and can execute the full
ingestion process:

```bash
docker compose --profile ingestion run ingestion
```

The command creates these local files:

```text
artifacts/dane-agent-data-v1.tar.gz
artifacts/dane-agent-data-v1.tar.gz.sha256
artifacts/dane-agent-data-v1.manifest.json
```

The ingestion profile does not upload anything.

## 3. Publish explicitly to Hugging Face

First create the public Dataset repository in the Hugging Face web interface.
Then set `HF_DATASET_REPOSITORY` and a write token in `HF_TOKEN` inside `.env`.

```bash
docker compose --profile publish run publisher
```

This uploads the three artifact files to
`releases/<DATA_VERSION>/` in that repository. Publishing is never part of
`docker compose up`.

## 4. Run the application

On any computer with Docker and a configured `.env`:

```bash
docker compose up --build
```

`data-init` downloads and validates the selected artifact into the
`knowledge_data` volume. Once it succeeds, `app` starts at
`http://localhost:8501`. The application reads the knowledge volume as
read-only and writes monitoring history to `observability_data`.

Use `docker compose down` to stop the stack while preserving data and monitoring
history. Use `docker compose down -v` only when you intentionally want to delete
the downloaded data and persisted interaction history.

To force a fresh data download for the same version, remove the data volume and
start again:

```bash
docker compose down -v
docker compose up --build
```
