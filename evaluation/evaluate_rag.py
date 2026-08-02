"""Evaluación offline del retrieval híbrido E5 + BM25.

Ejemplo:
    uv run python -m evaluation.evaluate_rag

No realiza llamadas a OpenAI. El resultado se escribe en
``evaluation/results/rag_retrieval_evaluation.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.rag_helper import (
    HybridRetriever,
    cross_validate_weights,
    evaluate_rankings,
    load_jsonl,
    retrieval_diagnostics,
)


EVALUATION_DIR = Path("evaluation")
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset_rag.jsonl"
RESULTS_PATH = EVALUATION_DIR / "results" / "rag_retrieval_evaluation.json"
TOP_K = 5
DENSE_WEIGHTS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def main() -> None:
    records = load_jsonl(GOLDEN_DATASET_PATH)
    duplicate_ids = {record["id"] for record in records if sum(
        candidate["id"] == record["id"] for candidate in records
    ) > 1}
    if duplicate_ids:
        raise ValueError(f"IDs duplicados en el golden dataset: {sorted(duplicate_ids)}")

    retriever = HybridRetriever()
    missing_chunk_ids = [
        record["id"]
        for record in records
        if record["chunk_id"] not in retriever.chunk_positions
    ]
    if missing_chunk_ids:
        raise ValueError(f"chunk_id inexistente para: {missing_chunk_ids}")

    query_embeddings = retriever.encode_queries(record["question"] for record in records)
    rankings_by_weight = {}
    grid_search = []

    for dense_weight in DENSE_WEIGHTS:
        rankings = {
            record["id"]: retriever.rank(
                question=record["question"],
                question_embedding=query_embedding,
                dense_weight=dense_weight,
                top_k=TOP_K,
            )
            for record, query_embedding in zip(records, query_embeddings, strict=True)
        }
        rankings_by_weight[dense_weight] = rankings
        metrics = evaluate_rankings(records, rankings)
        grid_search.append(
            {
                "dense_weight": dense_weight,
                "bm25_weight": 1 - dense_weight,
                "hit_rate": metrics["hit_rate"],
                "mrr": metrics["mrr"],
            }
        )

    best = max(grid_search, key=lambda result: (result["mrr"], result["hit_rate"]))
    selected_metrics = evaluate_rankings(records, rankings_by_weight[best["dense_weight"]])
    results = {
        "dataset": str(GOLDEN_DATASET_PATH),
        "query_count": len(records),
        "top_k": TOP_K,
        "metric_definitions": {
            "hit_rate": "Proporción de preguntas cuyo chunk relevante aparece en los primeros k resultados.",
            "mrr": "Promedio del inverso del rango del primer chunk relevante en los primeros k resultados.",
        },
        "grid_search": grid_search,
        "best_weight_full_dataset": best,
        "cross_validation": cross_validate_weights(records, rankings_by_weight, folds=5),
        "selected_weight_details": selected_metrics,
        "selected_weight_diagnostics": retrieval_diagnostics(selected_metrics, top_k=TOP_K),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Golden dataset: {len(records)} preguntas | top_k={TOP_K}")
    print("\nResultados por peso (E5 / BM25):")
    for result in grid_search:
        print(
            f"  {result['dense_weight']:.1f} / {result['bm25_weight']:.1f}"
            f" | Hit Rate: {result['hit_rate']:.3f}"
            f" | MRR: {result['mrr']:.3f}"
        )
    print(
        "\nMejor peso en el conjunto completo: "
        f"E5={best['dense_weight']:.1f}, BM25={best['bm25_weight']:.1f}"
    )
    nested = results["cross_validation"]["nested_cross_validation"]
    print(
        "Validación cruzada anidada (5 folds): "
        f"Hit Rate={nested['mean_hit_rate']:.3f}, MRR={nested['mean_mrr']:.3f}"
    )
    diagnostics = results["selected_weight_diagnostics"]
    print(f"Distribución de rangos: {diagnostics['rank_distribution']}")
    print(
        "Resumen de scores del chunk correcto: "
        f"{diagnostics['correct_chunk_score_summary']}"
    )
    print(f"\nResultados detallados: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
