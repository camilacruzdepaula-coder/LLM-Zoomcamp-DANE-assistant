"""Evaluacion automatica de relevancia para lotes de interacciones reales."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from toyaikit.pricing import PricingConfig


JUDGE_MODEL = "gpt-5-mini"
JUDGE_BATCH_SIZE = 3
JUDGE_INSTRUCTIONS = """
Eres un evaluador de relevancia para un asistente estadistico del DANE.

Para cada interaccion, decide si la respuesta atiende de forma directa la pregunta
del usuario. Marca 1 si es relevante y -1 si es irrelevante, incompleta de forma
material, evasiva o no responde a lo pedido. No evalues estilo ni supongas datos
que no aparecen en la pregunta y respuesta. Escribe una razon breve en espanol.
Devuelve una evaluacion para cada interaction_id recibido y no inventes ids.
""".strip()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "interaction_id": {"type": "string"},
                    "score": {"type": "integer", "enum": [-1, 1]},
                    "reason": {"type": "string"},
                },
                "required": ["interaction_id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assessments"],
    "additionalProperties": False,
}


@dataclass
class JudgeAssessment:
    interaction_id: str
    score: int
    reason: str


@dataclass
class JudgeBatchResult:
    assessments: list[JudgeAssessment]
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    response_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["assessments"] = [asdict(item) for item in self.assessments]
        return result


class DaneRelevanceJudge:
    def __init__(self, client: Any | None = None) -> None:
        load_dotenv()
        self.client = client or OpenAI(timeout=60, max_retries=0)
        self.pricing = PricingConfig()

    def evaluate(self, interactions: list[dict]) -> JudgeBatchResult:
        if not interactions:
            raise ValueError("Se necesita al menos una interaccion para evaluar")

        payload = [
            {
                "interaction_id": interaction["interaction_id"],
                "question": interaction["question"],
                "answer": interaction["answer"],
            }
            for interaction in interactions
        ]
        started_at = perf_counter()
        response = self.client.responses.create(
            model=JUDGE_MODEL,
            instructions=JUDGE_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "relevance_assessments",
                    "description": "Evaluaciones de relevancia por interaccion.",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                },
                "verbosity": "low",
            },
        )
        parsed = json.loads(response.output_text)
        allowed_ids = {interaction["interaction_id"] for interaction in interactions}
        assessments = [
            JudgeAssessment(
                interaction_id=item["interaction_id"],
                score=item["score"],
                reason=item["reason"].strip(),
            )
            for item in parsed["assessments"]
            if item["interaction_id"] in allowed_ids and item["score"] in {-1, 1}
        ]
        if {item.interaction_id for item in assessments} != allowed_ids:
            raise ValueError("El judge no devolvio una evaluacion para cada interaccion")

        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        pricing = self.pricing.calculate_cost(JUDGE_MODEL, input_tokens, output_tokens)
        total_cost = 0.0 if pricing is None else float(pricing.total_cost)
        return JudgeBatchResult(
            assessments=assessments,
            model=JUDGE_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            total_cost=total_cost,
            response_time_seconds=perf_counter() - started_at,
        )
