"""Estructuras de observabilidad para cada interacción con el agente."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class AgentRunRecord:
    question: str
    answer: str
    model: str
    tool_calls: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response_time_seconds: float
    total_cost: float
    error: str | None = None
    interaction_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["timestamp"] = self.timestamp.isoformat()
        return record
