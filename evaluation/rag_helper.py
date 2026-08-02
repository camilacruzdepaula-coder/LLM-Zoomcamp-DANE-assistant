"""Utilidades locales para evaluar la recuperación híbrida del proyecto DANE.

No llama a OpenAI. Carga los embeddings E5 y el índice BM25 ya construidos,
reproduce el ranking híbrido de runtime y calcula Hit Rate y MRR.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from dane_assistant.config import RAG_INDEX_DIR


INDEX_DIR = RAG_INDEX_DIR
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
BM25_PATH = INDEX_DIR / "bm25_index.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    minimum = scores.min()
    maximum = scores.max()
    if math.isclose(float(minimum), float(maximum)):
        return np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


class HybridRetriever:
    """Ranking E5 + BM25 parametrizable, sin filtro de umbral.

    El umbral se aplica al contexto que llega al LLM. Para Hit Rate y MRR se
    evalúa el ranking completo, por lo que se recuperan siempre los primeros k.
    """

    def __init__(self) -> None:
        self.chunks = load_jsonl(CHUNKS_PATH)
        self.embeddings = np.load(EMBEDDINGS_PATH)
        with BM25_PATH.open("r", encoding="utf-8") as file:
            self.bm25_index = json.load(file)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError("La cantidad de chunks no coincide con los embeddings.")
        if self.bm25_index["document_count"] != len(self.chunks):
            raise ValueError("El índice BM25 no coincide con los chunks.")

        self.chunk_positions = {
            chunk["chunk_id"]: position for position, chunk in enumerate(self.chunks)
        }
        self.model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

    def bm25_scores(self, question: str) -> np.ndarray:
        scores = np.zeros(self.bm25_index["document_count"], dtype=np.float32)
        document_count = self.bm25_index["document_count"]
        average_length = self.bm25_index["average_document_length"]
        document_lengths = self.bm25_index["document_lengths"]
        postings = self.bm25_index["postings"]
        k1 = self.bm25_index["k1"]
        b = self.bm25_index["b"]

        for term in set(tokenize(question)):
            term_postings = postings.get(term, [])
            if not term_postings:
                continue

            document_frequency = len(term_postings)
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document_index, term_frequency in term_postings:
                length_ratio = document_lengths[document_index] / average_length
                denominator = term_frequency + k1 * (1 - b + b * length_ratio)
                scores[document_index] += (
                    inverse_document_frequency * term_frequency * (k1 + 1) / denominator
                )
        return scores

    def encode_queries(self, questions: Iterable[str]) -> np.ndarray:
        prefixed = [f"query: {question}" for question in questions]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def rank(
        self,
        question: str,
        question_embedding: np.ndarray,
        dense_weight: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight debe estar entre 0 y 1.")

        dense_scores = self.embeddings @ question_embedding
        lexical_scores = self.bm25_scores(question)
        hybrid_scores = (
            dense_weight * normalize_scores(dense_scores)
            + (1 - dense_weight) * normalize_scores(lexical_scores)
        )
        positions = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            {
                "chunk_id": self.chunks[position]["chunk_id"],
                "rank": rank,
                "score": float(hybrid_scores[position]),
                "dense_score": float(dense_scores[position]),
                "bm25_score": float(lexical_scores[position]),
            }
            for rank, position in enumerate(positions, start=1)
        ]


def evaluate_rankings(
    records: list[dict[str, Any]],
    rankings: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Calcula Hit Rate y MRR para un conjunto de rankings ya generado."""
    details = []
    reciprocal_ranks = []

    for record in records:
        relevant_ids = {record["chunk_id"]}
        ranking = rankings[record["id"]]
        relevant_result = next(
            (result for result in ranking if result["chunk_id"] in relevant_ids),
            None,
        )
        rank = relevant_result["rank"] if relevant_result else None
        reciprocal_rank = 1 / rank if rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        details.append(
            {
                "id": record["id"],
                "question": record["question"],
                "source": record["source"],
                "hit": rank is not None,
                "rank": rank,
                "reciprocal_rank": reciprocal_rank,
                "correct_chunk_score": relevant_result["score"] if relevant_result else None,
                "retrieved_chunk_ids": [result["chunk_id"] for result in ranking],
            }
        )

    return {
        "query_count": len(records),
        "hit_rate": float(np.mean([detail["hit"] for detail in details])),
        "mrr": float(np.mean(reciprocal_ranks)),
        "details": details,
    }


def retrieval_diagnostics(metrics: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Resume el rango y score híbrido alcanzado por los chunks correctos."""
    details = metrics["details"]
    rank_distribution = {
        str(rank): sum(detail["rank"] == rank for detail in details)
        for rank in range(1, top_k + 1)
    }
    rank_distribution["not_retrieved"] = sum(
        detail["rank"] is None for detail in details
    )
    correct_scores = [
        detail["correct_chunk_score"]
        for detail in details
        if detail["correct_chunk_score"] is not None
    ]
    score_summary = {}
    if correct_scores:
        score_array = np.asarray(correct_scores)
        score_summary = {
            "count": len(correct_scores),
            "min": float(np.min(score_array)),
            "p25": float(np.percentile(score_array, 25)),
            "median": float(np.median(score_array)),
            "mean": float(np.mean(score_array)),
            "p75": float(np.percentile(score_array, 75)),
            "max": float(np.max(score_array)),
        }
    score_bins = {
        "[0.0, 0.2)": 0,
        "[0.2, 0.4)": 0,
        "[0.4, 0.6)": 0,
        "[0.6, 0.8)": 0,
        "[0.8, 1.0]": 0,
    }
    for score in correct_scores:
        if score < 0.2:
            score_bins["[0.0, 0.2)"] += 1
        elif score < 0.4:
            score_bins["[0.2, 0.4)"] += 1
        elif score < 0.6:
            score_bins["[0.4, 0.6)"] += 1
        elif score < 0.8:
            score_bins["[0.6, 0.8)"] += 1
        else:
            score_bins["[0.8, 1.0]"] += 1

    return {
        "rank_distribution": rank_distribution,
        "correct_chunk_score_summary": score_summary,
        "correct_chunk_score_bins": score_bins,
    }


def stratified_folds(records: list[dict[str, Any]], folds: int = 5) -> list[list[dict[str, Any]]]:
    """Divide las consultas preservando la distribución por fuente documental."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)

    result: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for source_index, source_records in enumerate(by_source.values()):
        for index, record in enumerate(source_records):
            # Rota el inicio de cada fuente para repartir los remanentes
            # entre folds y evitar particiones con tamaños desiguales.
            result[(index + source_index) % folds].append(record)
    return result


def cross_validate_weights(
    records: list[dict[str, Any]],
    rankings_by_weight: dict[float, dict[str, list[dict[str, Any]]]],
    folds: int = 5,
) -> dict[str, Any]:
    """Mide cada peso en folds estratificados y hace selección anidada.

    La combinación E5/BM25 no se entrena; la validación cruzada cuantifica la
    estabilidad de las métricas y evita elegir un peso por una sola partición.
    """
    fold_records = stratified_folds(records, folds)
    weights = sorted(rankings_by_weight)
    summaries = []
    nested_folds = []

    for weight in weights:
        fold_metrics = [
            evaluate_rankings(test_records, rankings_by_weight[weight])
            for test_records in fold_records
        ]
        summaries.append(
            {
                "dense_weight": weight,
                "bm25_weight": 1 - weight,
                "mean_hit_rate": float(np.mean([metric["hit_rate"] for metric in fold_metrics])),
                "std_hit_rate": float(np.std([metric["hit_rate"] for metric in fold_metrics])),
                "mean_mrr": float(np.mean([metric["mrr"] for metric in fold_metrics])),
                "std_mrr": float(np.std([metric["mrr"] for metric in fold_metrics])),
            }
        )

    for fold_index, test_records in enumerate(fold_records, start=1):
        test_ids = {record["id"] for record in test_records}
        train_records = [record for record in records if record["id"] not in test_ids]
        selected_weight = max(
            weights,
            key=lambda weight: (
                evaluate_rankings(train_records, rankings_by_weight[weight])["mrr"],
                evaluate_rankings(train_records, rankings_by_weight[weight])["hit_rate"],
            ),
        )
        test_metrics = evaluate_rankings(test_records, rankings_by_weight[selected_weight])
        nested_folds.append(
            {
                "fold": fold_index,
                "selected_dense_weight": selected_weight,
                "selected_bm25_weight": 1 - selected_weight,
                "hit_rate": test_metrics["hit_rate"],
                "mrr": test_metrics["mrr"],
                "query_count": test_metrics["query_count"],
            }
        )

    return {
        "per_weight": summaries,
        "nested_cross_validation": {
            "folds": nested_folds,
            "mean_hit_rate": float(np.mean([fold["hit_rate"] for fold in nested_folds])),
            "mean_mrr": float(np.mean([fold["mrr"] for fold in nested_folds])),
        },
    }
