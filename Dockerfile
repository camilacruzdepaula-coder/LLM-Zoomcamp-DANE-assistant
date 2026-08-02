FROM ghcr.io/astral-sh/uv:0.11.29 AS uv


FROM python:3.13-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv /uv /uvx /bin/


FROM python-base AS app

COPY . ./
RUN uv sync --no-dev --no-install-project \
    && .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')" \
    && mkdir --parents /app/app_data

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]


FROM python-base AS ingestion

COPY . ./
RUN uv sync --no-dev --no-install-project --group ingestion

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "dane_assistant.ingestion.pipeline"]


FROM python-base AS publisher

COPY . ./
RUN uv sync --no-dev --no-install-project --group publish

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "dane_assistant.ingestion.publish"]


FROM python-base AS data-init

COPY dane_assistant/ingestion/bootstrap.py /bootstrap.py

ENTRYPOINT ["python", "/bootstrap.py"]
