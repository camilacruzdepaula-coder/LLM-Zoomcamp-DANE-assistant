"""Genera evaluation/evaluation.ipynb con dos partes:

Parte 1 — RAG: lee rag_retrieval_evaluation.json y bake los outputs (sin costo).
Parte 2 — Agente: celdas ejecutables que reanudan desde agent_evaluation.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_RESULTS = ROOT / "evaluation" / "results" / "rag_retrieval_evaluation.json"
NOTEBOOK = ROOT / "evaluation" / "evaluation.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_baked(source, output):
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [{"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}],
    }


def code_blank(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
    }


def build_rag_cells(results):
    best = results["best_weight_full_dataset"]
    validation = results["cross_validation"]["nested_cross_validation"]
    diagnostics = results["selected_weight_diagnostics"]

    grid_output = "\n".join(
        f"E5={row['dense_weight']:.1f} | BM25={row['bm25_weight']:.1f} | "
        f"Hit Rate={row['hit_rate']:.3f} | MRR={row['mrr']:.3f}"
        for row in results["grid_search"]
    ) + "\n"

    rank_output = "\n".join(
        f"{rank}: {count}" for rank, count in diagnostics["rank_distribution"].items()
    ) + "\n"

    score_summary = diagnostics["correct_chunk_score_summary"]
    score_bins = diagnostics["correct_chunk_score_bins"]
    score_output = (
        "\n".join(
            f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}"
            for name, value in score_summary.items()
        )
        + "\n\n"
        + "\n".join(f"{interval}: {count}" for interval, count in score_bins.items())
        + "\n"
    )

    return [
        md(
            "## Parte 1: Evaluación del RAG híbrido\n\n"
            "Golden dataset: 50 preguntas documentales. Lee `rag_retrieval_evaluation.json` — sin costo.\n\n"
            "**Métricas**\n"
            "- **Hit Rate@5**: el chunk correcto aparece en los cinco primeros resultados.\n"
            "- **MRR@5**: premia que el chunk correcto aparezca en posiciones más altas."
        ),
        md("### Grid search — combinaciones de pesos E5 / BM25"),
        code_baked(
            "from pathlib import Path\nimport json\n\n"
            "root = Path.cwd()\n"
            "if not (root / 'pyproject.toml').exists():\n"
            "    root = root.parent\n\n"
            "rag_path = root / 'evaluation' / 'results' / 'rag_retrieval_evaluation.json'\n"
            "rag = json.loads(rag_path.read_text(encoding='utf-8'))\n\n"
            "for row in rag['grid_search']:\n"
            "    print(f\"E5={row['dense_weight']:.1f} | BM25={row['bm25_weight']:.1f} | \"\n"
            "          f\"Hit Rate={row['hit_rate']:.3f} | MRR={row['mrr']:.3f}\")\n",
            grid_output,
        ),
        md("### Posición del chunk correcto en top_k=5"),
        code_baked(
            "diagnostics = rag['selected_weight_diagnostics']\n\n"
            "for rank, count in diagnostics['rank_distribution'].items():\n"
            "    print(f'{rank}: {count}')\n",
            rank_output,
        ),
        md("### Score híbrido del chunk correcto recuperado"),
        code_baked(
            "summary = diagnostics['correct_chunk_score_summary']\n"
            "bins = diagnostics['correct_chunk_score_bins']\n\n"
            "for name, value in summary.items():\n"
            "    print(f'{name}: {value:.3f}' if isinstance(value, float) else f'{name}: {value}')\n\n"
            "for interval, count in bins.items():\n"
            "    print(f'{interval}: {count}')\n",
            score_output,
        ),
        md(
            "### Conclusión RAG\n\n"
            f"Usar **E5 = {best['dense_weight']:.0%}** y **BM25 = {best['bm25_weight']:.0%}**. "
            f"Hit Rate@5: **{best['hit_rate']:.3f}**. MRR@5: **{best['mrr']:.3f}**. "
            f"Validación cruzada anidada (5 folds): Hit Rate **{validation['mean_hit_rate']:.3f}**, "
            f"MRR **{validation['mean_mrr']:.3f}**."
        ),
    ]


def build_agent_cells():
    setup_code = """\
import sys
import os
import json
from pathlib import Path
from collections import defaultdict

root = Path.cwd()
if not (root / 'pyproject.toml').exists():
    root = root.parent
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from evaluation.agent_helper import (
    load_jsonl, validate_dataset, run_agent, run_judge, save_results, cost_summary
)

GOLDEN = root / 'evaluation' / 'golden_dataset_agent.jsonl'
RESULTS_PATH = root / 'evaluation' / 'results' / 'agent_evaluation.json'

golden = load_jsonl(GOLDEN)
existing = json.loads(RESULTS_PATH.read_text(encoding='utf-8')) if RESULTS_PATH.exists() else []
done_ids = {r['id'] for r in existing}
pending = [r for r in golden if r['id'] not in done_ids]

report = validate_dataset(golden)
print(f"Dataset válido: {report['valid']}")
print(f"Total preguntas: {report['record_count']}")
print(f"Categorías: {report['categories']}")
print(f"Ya ejecutadas: {len(existing)}")
print(f"Pendientes: {len(pending)}")
if pending:
    print("IDs pendientes: " + ", ".join(r['id'] for r in pending))
"""

    loop_code = """\
all_results = list(existing)

for record in pending:
    print(f"Ejecutando {record['id']} ({record['category']})...")
    result = run_agent([record])[0]
    result = run_judge([result])[0]
    all_results.append(result)
    save_results(all_results, RESULTS_PATH)
    print(f"  guardado | agente ${result['agent_cost']:.5f} | judge ${result['judge_cost']:.5f}")

if not pending:
    print("No hay preguntas pendientes.")
else:
    print(f"\\nListo. Total guardado: {len(all_results)} preguntas.")
"""

    summary_code = """\
import sys
import json
from pathlib import Path
from collections import defaultdict

root = Path.cwd()
if not (root / 'pyproject.toml').exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from evaluation.agent_helper import cost_summary

RESULTS_PATH = root / 'evaluation' / 'results' / 'agent_evaluation.json'
results = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))
judged = [r for r in results if 'judge' in r]

total = len(judged)
answer_good = sum(r['judge']['answer_score'] == 'good' for r in judged)
trajectory_good = sum(r['judge']['trajectory_score'] == 'good' for r in judged)
sequence_good = sum(r['expected_sequence_used'] for r in judged)
costs = cost_summary(judged)

print(f"Respuestas buenas:   {answer_good}/{total}")
print(f"Trayectorias buenas: {trajectory_good}/{total}")
print(f"Secuencia esperada:  {sequence_good}/{total}")
print()
print(f"Costo agente: ${costs['agent_cost']:.6f}")
print(f"Costo judge:  ${costs['judge_cost']:.6f}")
print(f"Costo total:  ${costs['total_cost']:.6f}")
print()

categories = defaultdict(list)
for r in judged:
    categories[r['category']].append(r)

for cat, rows in sorted(categories.items()):
    a = sum(r['judge']['answer_score'] == 'good' for r in rows)
    t = sum(r['judge']['trajectory_score'] == 'good' for r in rows)
    s = sum(r['expected_sequence_used'] for r in rows)
    print(f"{cat}: respuestas {a}/{len(rows)} | trayectorias {t}/{len(rows)} | secuencia {s}/{len(rows)}")
"""

    bitacora_code = """\
import json
from pathlib import Path

root = Path.cwd()
if not (root / 'pyproject.toml').exists():
    root = root.parent

results_dir = root / 'evaluation' / 'results'

KNOWN_ORDER = [
    'agent_evaluation_baseline.json',
    'agent_evaluation_v1_sin_stopping_rules.json',
    'agent_evaluation_v2_con_stopping_rules_mini.json',
    'agent_evaluation.json',
]

all_files = list(results_dir.glob('agent_evaluation*.json'))
ordered = [results_dir / n for n in KNOWN_ORDER if (results_dir / n).exists()]
for f in sorted(all_files):
    if f not in ordered:
        ordered.append(f)

print(f"{'Corrida':<48} {'Resp':>5} {'Tray':>5} {'Seq':>5} {'Costo':>10}")
print("-" * 78)
for f in ordered:
    data = json.loads(f.read_text(encoding='utf-8'))
    judged = [r for r in data if 'judge' in r and not r.get('category', '').endswith('_fallback')]
    if not judged:
        continue
    n = len(judged)
    ans  = sum(r['judge']['answer_score'] == 'good' for r in judged)
    traj = sum(r['judge']['trajectory_score'] == 'good' for r in judged)
    seq  = sum(r['expected_sequence_used'] for r in judged)
    cost = sum(r.get('total_cost', 0) for r in judged)
    print(f"{f.stem[:47]:<48} {ans}/{n}  {traj}/{n}  {seq}/{n}  ${cost:.4f}")
"""

    return [
        md(
            "---\n\n"
            "## Parte 2: Evaluación del agente\n\n"
            "Dataset: 30 preguntas (document_search, statistic_direct, comparison_direct). "
            "Reanuda desde `agent_evaluation.json` — solo ejecuta las preguntas pendientes.\n\n"
            "**Ejecutar con el entorno de uv. No correr en paralelo con `python -m evaluation.evaluate_agent`.**"
        ),
        md("### Celda 1 — Imports, rutas y estado del dataset"),
        code_blank(setup_code),
        md("### Celda 2 — Loop de evaluación (agente + judge por pregunta)"),
        code_blank(loop_code),
        md("### Celda 3 — Resumen de la corrida actual"),
        code_blank(summary_code),
        md(
            "### Celda 4 — Bitácora de corridas\n\n"
            "Compara todas las corridas guardadas en `evaluation/results/`. "
            "Los fallbacks se excluyen para mantener comparabilidad entre versiones."
        ),
        code_blank(bitacora_code),
    ]


def main():
    results = json.loads(RAG_RESULTS.read_text(encoding="utf-8"))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "cells": [
            md(
                "# Evaluación DANE — TIC e I+D\n\n"
                "## Índice\n"
                "1. [Parte 1: Evaluación del RAG híbrido](#parte-1-evaluación-del-rag-híbrido) — sin costo, resultados precalculados\n"
                "2. [Parte 2: Evaluación del agente](#parte-2-evaluación-del-agente) — ejecutar con kernel `.venv`\n\n"
                "---"
            ),
            *build_rag_cells(results),
            *build_agent_cells(),
        ],
    }

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Notebook generado: {NOTEBOOK}")


if __name__ == "__main__":
    main()
