"""Capa de aplicación para ejecutar y medir una interacción con el agente DANE."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from dane_assistant.agent.telemetry import cost_value, extract_tool_calls
from dane_assistant.monitoring.metrics import AgentRunRecord


@dataclass
class AgentRunResult:
    record: AgentRunRecord
    conversation_messages: list[Any]


class DaneAgentService:
    def __init__(self) -> None:
        from dane_assistant.agent.runtime import build_runner

        self.runner = build_runner()

    def ask(
        self, question: str, conversation_messages: list[Any] | None = None
    ) -> AgentRunResult:
        started_at = perf_counter()
        try:
            response = self.runner.loop(
                prompt=question,
                previous_messages=conversation_messages,
            )
        except Exception as error:
            record = AgentRunRecord(
                question=question,
                answer="",
                model=self.runner.llm_client.model,
                tool_calls=[],
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                response_time_seconds=perf_counter() - started_at,
                total_cost=0.0,
                error=f"{type(error).__name__}: {error}",
            )
            return AgentRunResult(record=record, conversation_messages=[])

        record = AgentRunRecord(
            question=question,
            answer=response.last_message,
            model=response.tokens.model,
            tool_calls=extract_tool_calls(response.all_messages),
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.output_tokens,
            total_tokens=response.tokens.input_tokens + response.tokens.output_tokens,
            response_time_seconds=perf_counter() - started_at,
            total_cost=cost_value(response.cost),
        )
        return AgentRunResult(
            record=record,
            conversation_messages=response.all_messages,
        )
