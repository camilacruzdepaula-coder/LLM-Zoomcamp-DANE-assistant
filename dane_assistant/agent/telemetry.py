"""Utilidades para extraer telemetría de respuestas del runner del agente."""

from __future__ import annotations


def extract_tool_calls(messages: list) -> list[dict]:
    calls = []
    for message in messages:
        if isinstance(message, dict) or getattr(message, "type", None) != "function_call":
            continue
        calls.append({"name": message.name, "arguments": message.arguments})
    return calls


def cost_value(cost) -> float:
    if cost is None:
        return 0.0
    return float(getattr(cost, "total_cost", cost))
