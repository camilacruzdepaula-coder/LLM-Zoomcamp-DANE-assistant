"""Persistencia SQLite y consultas agregadas para la observabilidad del agente."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from dane_assistant.monitoring.metrics import AgentRunRecord


DEFAULT_DATABASE_PATH = Path("app_data/observability.db")


class ObservabilityStore:
    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    tool_count INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time_seconds REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    error TEXT,
                    feedback INTEGER CHECK (feedback IN (-1, 1)),
                    feedback_comment TEXT,
                    feedback_timestamp TEXT,
                    llm_judge_score INTEGER CHECK (llm_judge_score IN (-1, 1)),
                    llm_judge_reason TEXT,
                    llm_judge_model TEXT,
                    llm_judge_timestamp TEXT,
                    llm_judge_error TEXT,
                    llm_judge_input_tokens INTEGER,
                    llm_judge_output_tokens INTEGER,
                    llm_judge_total_tokens INTEGER,
                    llm_judge_cost REAL,
                    llm_judge_latency_seconds REAL
                );

                CREATE INDEX IF NOT EXISTS interactions_timestamp_idx
                    ON interactions(timestamp);
                CREATE INDEX IF NOT EXISTS interactions_feedback_idx
                    ON interactions(feedback);
                """
            )
            self._migrate_judge_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS interactions_llm_judge_idx
                ON interactions(llm_judge_score)
                """
            )

    @staticmethod
    def _migrate_judge_columns(connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(interactions)").fetchall()
        }
        columns = {
            "llm_judge_score": "INTEGER CHECK (llm_judge_score IN (-1, 1))",
            "llm_judge_reason": "TEXT",
            "llm_judge_model": "TEXT",
            "llm_judge_timestamp": "TEXT",
            "llm_judge_error": "TEXT",
            "llm_judge_input_tokens": "INTEGER",
            "llm_judge_output_tokens": "INTEGER",
            "llm_judge_total_tokens": "INTEGER",
            "llm_judge_cost": "REAL",
            "llm_judge_latency_seconds": "REAL",
        }
        for name, definition in columns.items():
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE interactions ADD COLUMN {name} {definition}")

    def save_interaction(self, record: AgentRunRecord) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    interaction_id, timestamp, question, answer, model,
                    tool_calls_json, tool_count, input_tokens, output_tokens,
                    total_tokens, response_time_seconds, total_cost, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.interaction_id,
                    record.timestamp.isoformat(),
                    record.question,
                    record.answer,
                    record.model,
                    json.dumps(record.tool_calls, ensure_ascii=False),
                    len(record.tool_calls),
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.response_time_seconds,
                    record.total_cost,
                    record.error,
                ),
            )

    def save_feedback(
        self, interaction_id: str, feedback: int, comment: str = ""
    ) -> None:
        if feedback not in {-1, 1}:
            raise ValueError("feedback debe ser -1 o 1")
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE interactions
                SET feedback = ?, feedback_comment = ?, feedback_timestamp = datetime('now')
                WHERE interaction_id = ?
                """,
                (feedback, comment.strip(), interaction_id),
            )

    def list_interactions(self, limit: int = 100) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_llm_judge_interactions(self, limit: int) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT interaction_id, question, answer
                FROM interactions
                WHERE llm_judge_score IS NULL
                  AND error IS NULL
                  AND trim(answer) != ''
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_llm_judgements(
        self,
        assessments: list[dict],
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        total_cost: float,
        response_time_seconds: float,
    ) -> None:
        if not assessments:
            return
        assessment_count = len(assessments)
        cost_per_assessment = total_cost / assessment_count
        latency_per_assessment = response_time_seconds / len(assessments)

        def allocated_tokens(total: int, index: int) -> int:
            base, remainder = divmod(total, assessment_count)
            return base + (1 if index < remainder else 0)

        with self.connect() as connection:
            connection.executemany(
                """
                UPDATE interactions
                SET
                    llm_judge_score = ?,
                    llm_judge_reason = ?,
                    llm_judge_model = ?,
                    llm_judge_timestamp = datetime('now'),
                    llm_judge_error = NULL,
                    llm_judge_input_tokens = ?,
                    llm_judge_output_tokens = ?,
                    llm_judge_total_tokens = ?,
                    llm_judge_cost = ?,
                    llm_judge_latency_seconds = ?
                WHERE interaction_id = ?
                """,
                [
                    (
                        assessment["score"],
                        assessment["reason"],
                        model,
                        allocated_tokens(input_tokens, index),
                        allocated_tokens(output_tokens, index),
                        allocated_tokens(total_tokens, index),
                        cost_per_assessment,
                        latency_per_assessment,
                        assessment["interaction_id"],
                    )
                    for index, assessment in enumerate(assessments)
                ],
            )

    def save_llm_judge_error(self, interaction_ids: list[str], error: str) -> None:
        if not interaction_ids:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                UPDATE interactions
                SET llm_judge_error = ?
                WHERE interaction_id = ?
                """,
                [(error, interaction_id) for interaction_id in interaction_ids],
            )

    def summary(self) -> dict[str, float | int | None]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS interaction_count,
                    COALESCE(AVG(response_time_seconds), 0) AS average_latency,
                    COALESCE(SUM(total_cost), 0) AS total_cost,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count,
                    SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) AS positive_feedback,
                    SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) AS negative_feedback,
                    SUM(CASE WHEN llm_judge_score = 1 THEN 1 ELSE 0 END) AS llm_relevant,
                    SUM(CASE WHEN llm_judge_score = -1 THEN 1 ELSE 0 END) AS llm_not_relevant,
                    COALESCE(SUM(llm_judge_input_tokens), 0) AS llm_judge_input_tokens,
                    COALESCE(SUM(llm_judge_output_tokens), 0) AS llm_judge_output_tokens,
                    COALESCE(SUM(llm_judge_total_tokens), 0) AS llm_judge_total_tokens,
                    COALESCE(SUM(llm_judge_cost), 0) AS llm_judge_cost
                FROM interactions
                """
            ).fetchone()
        feedback_count = (row["positive_feedback"] or 0) + (row["negative_feedback"] or 0)
        llm_judged_count = (row["llm_relevant"] or 0) + (row["llm_not_relevant"] or 0)
        approval_rate = (
            (row["positive_feedback"] or 0) / feedback_count if feedback_count else None
        )
        return {
            "interaction_count": row["interaction_count"],
            "average_latency": row["average_latency"],
            "total_cost": row["total_cost"],
            "total_tokens": row["total_tokens"],
            "error_count": row["error_count"],
            "positive_feedback": row["positive_feedback"] or 0,
            "negative_feedback": row["negative_feedback"] or 0,
            "approval_rate": approval_rate,
            "llm_relevant": row["llm_relevant"] or 0,
            "llm_not_relevant": row["llm_not_relevant"] or 0,
            "llm_judged_count": llm_judged_count,
            "llm_relevance_rate": (
                (row["llm_relevant"] or 0) / llm_judged_count
                if llm_judged_count
                else None
            ),
            "llm_judge_input_tokens": row["llm_judge_input_tokens"],
            "llm_judge_output_tokens": row["llm_judge_output_tokens"],
            "llm_judge_total_tokens": row["llm_judge_total_tokens"],
            "llm_judge_cost": row["llm_judge_cost"],
        }

    def daily_metrics(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    substr(timestamp, 1, 10) AS day,
                    COUNT(*) AS interactions,
                    AVG(response_time_seconds) AS average_latency,
                    SUM(total_cost) AS agent_cost,
                    SUM(COALESCE(llm_judge_cost, 0)) AS llm_judge_cost,
                    SUM(total_cost) + SUM(COALESCE(llm_judge_cost, 0)) AS total_cost,
                    SUM(total_tokens) AS agent_tokens,
                    SUM(COALESCE(llm_judge_total_tokens, 0)) AS llm_judge_tokens,
                    SUM(total_tokens) + SUM(COALESCE(llm_judge_total_tokens, 0))
                        AS total_tokens
                FROM interactions
                GROUP BY day
                ORDER BY day
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def tool_usage(self) -> list[dict]:
        counts: Counter[str] = Counter()
        with self.connect() as connection:
            rows = connection.execute("SELECT tool_calls_json FROM interactions").fetchall()
        for row in rows:
            for tool_call in json.loads(row["tool_calls_json"]):
                counts[tool_call["name"]] += 1
        return [
            {"tool": tool_name, "calls": count}
            for tool_name, count in counts.most_common()
        ]

    def feedback_breakdown(self) -> list[dict]:
        return [
            {"feedback": "Útil", "count": self.summary()["positive_feedback"]},
            {"feedback": "No útil", "count": self.summary()["negative_feedback"]},
        ]

    def llm_judge_breakdown(self) -> list[dict]:
        summary = self.summary()
        return [
            {"feedback": "Relevante", "count": summary["llm_relevant"]},
            {"feedback": "No relevante", "count": summary["llm_not_relevant"]},
        ]
