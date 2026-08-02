"""Funciones simples para preparar y registrar la evaluación del agente."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from dane_assistant.agent.telemetry import cost_value, extract_tool_calls


TOOL_NAMES = {
    "search_dane_knowledge_base",
    "lookup_official_statistic",
    "compare_official_statistics",
}
REQUIRED_FIELDS = {"id", "category", "question", "expected_answer", "expected_tool_sequence"}

JUDGE_PROMPT = """
Eres un evaluador experto de un agente estadístico del DANE.

Recibirás una pregunta, la respuesta esperada, la respuesta del agente, la
secuencia de tools esperada y las llamadas realizadas por el agente.

Evalúa la respuesta. Es buena si contiene la información esencial de la respuesta
esperada, usa cifras correctas cuando correspondan y no inventa información. Si
no había evidencia suficiente, solo es buena si responde exactamente: "No tengo
evidencia suficiente todavía para responder la pregunta."

Evalúa la trayectoria. La secuencia de tools esperada indica la ruta deseada. Para
cifras debe iniciar con lookup_official_statistic; para comparaciones con
compare_official_statistics. Cuando la secuencia esperada termina con
search_dane_knowledge_base, el fallback está justificado y no debe penalizarse.
Penaliza llamadas duplicadas, innecesarias o que no apoyen la respuesta.

Usa exclusivamente la información incluida en la evaluación.
""".strip()


class JudgeResult(BaseModel):
    answer_score: Literal["good", "bad"]
    trajectory_score: Literal["good", "bad"]
    answer_reasoning: str
    trajectory_reasoning: str


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def validate_dataset(records: list[dict]) -> dict:
    errors = []
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("Hay IDs duplicados.")

    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            errors.append(f"{record.get('id', '<sin id>')}: faltan {sorted(missing)}.")
        unknown_tools = set(record.get("expected_tool_sequence", [])) - TOOL_NAMES
        if unknown_tools:
            errors.append(f"{record['id']}: tools desconocidas {sorted(unknown_tools)}.")

    return {
        "valid": not errors,
        "record_count": len(records),
        "categories": dict(Counter(record["category"] for record in records)),
        "errors": errors,
    }


def evaluate_trajectory(record: dict, tool_calls: list[dict]) -> dict:
    expected = record["expected_tool_sequence"]
    actual = [call["name"] for call in tool_calls]
    return {
        "expected_tools": expected,
        "actual_tools": actual,
        "expected_sequence_used": actual[: len(expected)] == expected,
        "tool_call_count": len(actual),
        "duplicate_tool_calls": len(actual) != len(set(actual)),
    }


def build_agent_runner():
    from dane_assistant.agent.runtime import build_runner

    return build_runner()


def run_agent(records: list[dict], runner=None) -> list[dict]:
    """Ejecuta el agente. Esta es la única función que llama al modelo."""
    runner = runner or build_agent_runner()
    results = []
    for record in records:
        response = runner.loop(prompt=record["question"])
        tool_calls = extract_tool_calls(response.all_messages)
        agent_cost = cost_value(response.cost)
        trajectory = evaluate_trajectory(record, tool_calls)
        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "question": record["question"],
                "expected_answer": record["expected_answer"],
                "answer_agent": response.last_message,
                "tool_calls": tool_calls,
                "agent_cost": agent_cost,
                "judge_cost": 0.0,
                "total_cost": agent_cost,
                **trajectory,
            }
        )
    return results


def build_judge_runner():
    from dotenv import load_dotenv
    from openai import OpenAI
    from toyaikit.chat.runners import OpenAIResponsesRunner
    from toyaikit.llm import OpenAIClient

    load_dotenv()
    return OpenAIResponsesRunner(
        developer_prompt=JUDGE_PROMPT,
        llm_client=OpenAIClient(
            model="gpt-4o-mini",
            client=OpenAI(timeout=60, max_retries=0),
        ),
    )


def run_judge(results: list[dict], runner=None) -> list[dict]:
    """Evalúa respuestas y trayectorias. Esta función también llama al modelo."""
    runner = runner or build_judge_runner()
    for result in results:
        prompt = json.dumps(
            {
                "question": result["question"],
                "expected_answer": result["expected_answer"],
                "agent_answer": result["answer_agent"],
                "expected_tool_sequence": result["expected_tools"],
                "tool_calls": result["tool_calls"],
            },
            ensure_ascii=False,
        )
        response = runner.loop(prompt=prompt, output_format=JudgeResult)
        result["judge"] = response.last_message.model_dump()
        result["judge_cost"] = cost_value(response.cost)
        result["total_cost"] = result["agent_cost"] + result["judge_cost"]
    return results


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def cost_summary(results: list[dict]) -> dict:
    agent_cost = sum(result.get("agent_cost", 0.0) for result in results)
    judge_cost = sum(result.get("judge_cost", 0.0) for result in results)
    return {
        "agent_cost": agent_cost,
        "judge_cost": judge_cost,
        "total_cost": agent_cost + judge_cost,
    }
