# Project Criteria

This document maps the implementation to the LLM Zoomcamp project criteria. It
describes the evidence in this repository rather than assigning points.

## 1. Problem description

The problem statement, target users, scope, and limitations are described in the
[Objective](README.md#objective) and [Scope and limitations](README.md#scope-and-limitations)
sections of the README.

The project solves a focused information-access problem: users need reliable
answers about DANE's technology and innovation statistics without manually
searching across reports, methodological documents, and tabular annexes. The
agent is restricted to a documented subset of DANE operations, and its official
sources are listed in [Official sources](README.md#official-sources).

## 2. Retrieval flow

The application uses both a knowledge base and an LLM.

- The documentary knowledge base contains chunked DANE HTML and PDF material,
  E5 embeddings, and a BM25 index. The hybrid retrieval implementation is in
  [dane_assistant/rag/search.py](dane_assistant/rag/search.py).
- The statistical knowledge base is a SQLite database with full-text search,
  built from normalized DANE tabular annexes. Its query and comparison tools are
  in [dane_assistant/agent/runtime.py](dane_assistant/agent/runtime.py).
- The agent uses gpt-5-mini to select one of three tools:
  search_dane_knowledge_base, lookup_official_statistic, or
  compare_official_statistics. The system instructions require answers to be
  based only on tool evidence.
- The ingestion flow that creates the document index and SQLite database is in
  [dane_assistant/ingestion/pipeline.py](dane_assistant/ingestion/pipeline.py).

The full routing flow is illustrated in [How it works](README.md#how-it-works).

## 3. Retrieval evaluation

Retrieval evaluation uses the 50-question
[golden RAG dataset](evaluation/golden_dataset_rag.jsonl). Every record identifies
the relevant source chunk, allowing ranking quality to be measured without an
LLM judge.

[evaluation/evaluate_rag.py](evaluation/evaluate_rag.py) evaluates seven dense/lexical combinations:
E5 weights from 0.0 to 1.0, with the remaining weight assigned to BM25. It
reports Hit Rate@5 and MRR@5, selects the best combination on the full dataset,
and runs nested five-fold cross-validation.

The saved [retrieval results](evaluation/results/rag_retrieval_evaluation.json)
selected E5 0.5 and BM25 0.5, with Hit Rate@5 of 0.920 and MRR@5 of
0.722. Those weights are the ones configured in
[dane_assistant/rag/search.py](dane_assistant/rag/search.py).

## 4. LLM evaluation

The agent is evaluated with the 30-question
[golden agent dataset](evaluation/golden_dataset_agent.jsonl). It includes
document retrieval, direct-statistic, and direct-comparison questions, along
with expected answers and expected tool sequences.

The workflow in [evaluation/evaluate_agent.py](evaluation/evaluate_agent.py) runs the agent and then
uses a separate gpt-4o-mini judge to assess answer correctness and trajectory
quality. It also checks whether the expected tool sequence was used. The
evaluation notebook, [evaluation/evaluation.ipynb](evaluation/evaluation.ipynb),
shows both the RAG and agent workflows.

Several prompt and routing variants were evaluated and retained in
[evaluation/results/](evaluation/results/). Each iteration was informed by the
questions that failed in the preceding run.

| Run | Change evaluated | Good answers | Good trajectories | Expected sequence |
| --- | --- | ---: | ---: | ---: |
| agent_evaluation_baseline | Initial prompt with gpt-4o-mini. | 15/30 | 10/30 | 15/30 |
| agent_evaluation_v1_sin_stopping_rules | Prompt adjusted from baseline failure analysis, still using gpt-4o-mini. | 18/30 | 15/30 | 29/30 |
| agent_evaluation_v2_con_stopping_rules_mini | Prompt adjusted with explicit stopping rules, using gpt-4o-mini. | 17/30 | 17/30 | 29/30 |
| agent_evaluation | Further prompt and stopping-rule adjustments based on the failed questions, using gpt-4o-mini. | 21/30 | 20/30 | 28/30 |
| agent_evaluation_v3_routing | Prompt and routing adjustments based on the failed questions, using gpt-5-mini. | 23/30 | 28/30 | 30/30 |

The selected behavior is implemented in the current gpt-5-mini system prompt,
routing rules, stopping rules, and table-label matching logic in
[dane_assistant/agent/runtime.py](dane_assistant/agent/runtime.py). It is the
best evaluated configuration in this repository.

A larger GPT-5 model is a reasonable next experiment for improving answer
quality. It was not evaluated in this project because the iterative testing
required multiple full agent-and-judge runs, and the budget was reserved for
those prompt, routing, and stopping-rule experiments.

## 5. Interface

The user-facing interface is a Streamlit application in [app.py](app.py).

It provides:
- A Spanish chat designed for the official DANE statistics use case.
- Execution details for every answer: model, latency, tokens, estimated cost,
  and tools used.
- Positive and negative user feedback controls.
- A persisted Monitoring Dashboard.

The interaction steps are documented in the [User guide](README.md#user-guide).

## 6. Ingestion pipeline

The project has an automated ingestion pipeline implemented as a Prefect flow in
[dane_assistant/ingestion/pipeline.py](dane_assistant/ingestion/pipeline.py).

The flow:
1. Scrapes the allowed official DANE source pages and their linked documents.
2. Normalizes the tabular annexes.
3. Builds the SQLite statistics database and its full-text index.
4. Extracts and chunks documentary content.
5. Generates E5 embeddings and the BM25 index.
6. Validates both knowledge bases.
7. Packages the runtime files with a manifest and SHA-256 checksums.

The local execution instructions are in [INGESTION.md](INGESTION.md). Docker also
exposes the same flow as the explicit ingestion profile.

## 7. Monitoring

Monitoring combines persisted user feedback with a Streamlit dashboard.

[dane_assistant/monitoring/store.py](dane_assistant/monitoring/store.py) stores
questions, answers, tool calls, tokens, latency, costs, feedback, and LLM judge
metadata in SQLite. The app renders seven dashboard charts in [app.py](app.py):

1. Interactions by day.
2. Cost by day.
3. Average latency by day.
4. Total tokens by day.
5. Tool usage.
6. User feedback.
7. LLM judge feedback.

After every three successful interactions, the gpt-5-mini relevance judge in
[dane_assistant/monitoring/judge.py](dane_assistant/monitoring/judge.py) evaluates
the question-answer pairs. Its score, explanation, tokens, cost, and latency are
also persisted.

## 8. Containerization

The complete runtime is defined in
[docker-compose.yml](docker-compose.yml), with four services:

- data-init downloads and validates the public data artifact into a named
  volume.
- app runs the Streamlit interface with the knowledge volume mounted
  read-only and a separate persistent observability volume.
- ingestion is an explicit profile that builds a local versioned artifact.
- publisher is an explicit profile that uploads a validated artifact to a
  Hugging Face Dataset repository.

The multi-stage [Dockerfile](Dockerfile) provides a dedicated target for each
service. Operational commands and volume behavior are documented in
[DOCKER.md](DOCKER.md).

## 9. Reproducibility

The repository declares application dependencies and optional ingestion and
publication groups in [pyproject.toml](pyproject.toml). Run `uv lock` with access
to the package registry and commit the resulting `uv.lock` before distribution.

It also includes [.env.example](.env.example), the local Python setup in the
[README](README.md#user-guide), the automated data build instructions
in [INGESTION.md](INGESTION.md), and the Docker execution guide in
[DOCKER.md](DOCKER.md).

For a fresh machine, Docker Compose downloads the runtime artifact from the
configured public Hugging Face Dataset repository. Before external users can run
that path, the artifact must be built and published from the source computer as
described in [DOCKER.md](DOCKER.md). The repository and version are intentionally
configured through HF_DATASET_REPOSITORY and DATA_VERSION, rather than
hard-coded.


## Best practice: Hybrid search

The documentary retrieval path combines semantic and lexical retrieval rather
than relying on a single search method.

- **Dense retrieval:** the multilingual E5 model embeds the user question and
  compares it with the precomputed document-chunk embeddings using cosine
  similarity.
- **Lexical retrieval:** a BM25 inverted index retrieves chunks containing the
  terms used in the question.
- **Hybrid ranking:** both score distributions are normalized per query and
  combined into one ranking. The runtime implementation is in
  [dane_assistant/rag/search.py](dane_assistant/rag/search.py).

This design is particularly useful for the DANE corpus. Dense retrieval helps
with paraphrases and conceptual questions, while BM25 preserves exact terms such
as operation names, acronyms, indicators, sectors, and technical vocabulary.

The hybrid configuration was evaluated rather than selected arbitrarily. The
retrieval evaluation in [evaluation/evaluate_rag.py](evaluation/evaluate_rag.py)
tests seven E5/BM25 weight combinations against the 50-question golden dataset,
reports Hit Rate@5 and MRR@5, and includes nested five-fold cross-validation.
The selected runtime weights are E5 0.5 and BM25 0.5; they achieved Hit Rate@5
of 0.920 and MRR@5 of 0.722 in the saved evaluation results.

## Additional project work

The following components go beyond a minimal retrieve-and-answer application.
They are documented as potential additional-project evidence; any bonus decision
remains with the reviewer.

### Evidence-aware tool routing

The agent does not use one generic retrieval call for every question. It routes
requests to three evidence-specific tools:

1. Documentary hybrid retrieval for definitions, methodology, context, and
   rankings.
2. SQLite full-text lookup over normalized official tabular annexes for an exact
   value, year, category, and unit.
3. Compatibility-checked comparisons that only calculate a difference when both
   values come from the same official table, metric, and unit.

The agent prompt also includes routing and stopping rules. Once a tool returns
sufficient evidence, the agent must answer instead of making unsupported or
duplicated retrieval calls. The implementation is in
[dane_assistant/agent/runtime.py](dane_assistant/agent/runtime.py), and the
effect of those rules is measured across the evaluation runs described above.

### Integrity-checked data artifacts

The ingestion pipeline produces a versioned runtime artifact instead of relying
on an undocumented local data folder. It validates the SQLite database, the RAG
index, and the agreement between chunk, embedding, and BM25 document counts.
Each runtime file and the archive receive SHA-256 checksums.

At application startup, data-init downloads the selected artifact, verifies the
archive checksum and every packaged runtime file, and installs only the validated
statistics and rag directories into the Docker volume. The relevant implementation
is in [dane_assistant/ingestion/pipeline.py](dane_assistant/ingestion/pipeline.py)
and [dane_assistant/ingestion/bootstrap.py](dane_assistant/ingestion/bootstrap.py).

### Operational quality signals

In addition to recording standard interaction telemetry, the application keeps
both human and automated quality signals for each persisted interaction. Users
can mark an answer as useful or not useful. Every three successful interactions,
a separate LLM judge evaluates direct relevance and stores its score, explanation,
model, token usage, cost, and latency.

This makes it possible to compare user satisfaction with automated relevance,
identify costly or slow tool paths, and review the actual question and answer
behind a metric. The persistence layer is in
[dane_assistant/monitoring/store.py](dane_assistant/monitoring/store.py), the
judge is in [dane_assistant/monitoring/judge.py](dane_assistant/monitoring/judge.py),
and the dashboard is in [app.py](app.py).
