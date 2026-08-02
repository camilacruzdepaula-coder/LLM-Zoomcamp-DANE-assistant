# Ingestion Pipeline

`dane_assistant.ingestion.pipeline` orchestrates the Prefect flow that rebuilds
the agent knowledge base. It does not publish artifacts and does not require an
OpenAI API key.

## Flow

1. Downloads official DANE Technology and Innovation pages and annexes.
2. Normalizes tabular annexes and creates `statistics.db` with full-text search.
3. Extracts PDF and HTML content, normalizes documents, and creates chunks.
4. Generates E5 embeddings and the BM25 index.
5. Validates the consistency of the tabular database and RAG index.
6. Creates a local runtime artifact and its checksums.

## Run on the source machine

```bash
uv sync --group ingestion
uv run --group ingestion python -m dane_assistant.ingestion.pipeline --version v1
```

The output is written to `artifacts/`:

```text
dane-agent-data-v1.tar.gz
dane-agent-data-v1.tar.gz.sha256
dane-agent-data-v1.manifest.json
```

The compressed archive contains only the files required at runtime:

```text
statistics/statistics.db
rag/chunks.jsonl
rag/embeddings.npy
rag/bm25_index.json
```

The `manifest.json` file records the version, size, and SHA-256 checksum for
every file. Publish the artifact later from the selected source machine.

## Useful options

```bash
# Check output paths without downloading or processing data.
uv run --group ingestion python -m dane_assistant.ingestion.pipeline --version v1 --dry-run

# Replace a previous local build.
uv run --group ingestion python -m dane_assistant.ingestion.pipeline --version v2 --overwrite
```
