"""Ejecuta el agente DANE y su judge, guardando resultados por pregunta."""

import argparse
import json
from pathlib import Path

from evaluation.agent_helper import (
    cost_summary,
    load_jsonl,
    run_agent,
    run_judge,
    save_results,
    validate_dataset,
)


DATASET = Path("evaluation/golden_dataset_agent.jsonl")
OUTPUT = Path("evaluation/results/agent_evaluation.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(DATASET)
    report = validate_dataset(records)
    if not report["valid"]:
        raise ValueError(report["errors"])

    results = json.loads(OUTPUT.read_text(encoding="utf-8")) if args.resume and OUTPUT.exists() else []
    completed_ids = {result["id"] for result in results}
    pending = [record for record in records if record["id"] not in completed_ids]
    if args.limit:
        pending = pending[: args.limit]

    for index, record in enumerate(pending, start=1):
        print(f"Procesando {index}/{len(pending)}: {record['id']}", flush=True)
        result = run_agent([record])
        result = run_judge(result)
        results.extend(result)
        save_results(results, OUTPUT)

    costs = cost_summary(results)
    answer_good = sum(result["judge"]["answer_score"] == "good" for result in results)
    trajectory_good = sum(result["judge"]["trajectory_score"] == "good" for result in results)
    route_good = sum(result["expected_sequence_used"] for result in results)
    print(f"Preguntas: {len(results)}")
    print(f"Respuesta good: {answer_good}/{len(results)}")
    print(f"Trayectoria good: {trajectory_good}/{len(results)}")
    print(f"Secuencia esperada: {route_good}/{len(results)}")
    print(f"Costo agente: ${costs['agent_cost']:.6f}")
    print(f"Costo judge: ${costs['judge_cost']:.6f}")
    print(f"Costo total: ${costs['total_cost']:.6f}")
    print(f"Resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
