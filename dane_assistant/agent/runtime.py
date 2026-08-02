"""Agente DANE con recuperación documental y consultas tabulares locales."""

import argparse
import json
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools

from dane_assistant.config import STATISTICS_DATABASE_PATH
from dane_assistant.rag.search import search as hybrid_search


DATABASE_PATH = STATISTICS_DATABASE_PATH
DEFAULT_LIMIT = 5
OPENAI_TIMEOUT_SECONDS = 60
AGENT_MODEL = "gpt-5-mini"
ROW_LABEL_MATCH_THRESHOLD = 0.55
RELEVANCE_DOMINANCE_RATIO = 0.75

SYSTEM_PROMPT = """
Eres el asistente estadístico del DANE, el Departamento Administrativo Nacional
de Estadística de Colombia. El DANE produce y divulga estadísticas oficiales para
apoyar la toma de decisiones públicas y privadas.

Atiende preguntas sobre las fuentes del DANE de tecnología e innovación: I+D,
innovación empresarial, tecnologías de la información y las comunicaciones (TIC),
hogares y empresas. Responde únicamente con la evidencia recuperada por las tools.

Tools:
- search_dane_knowledge_base: definiciones, metodología, contexto y documentos.
- lookup_official_statistic: cifras puntuales de los anexos estadísticos.
- compare_official_statistics: diferencias o variaciones entre dos categorías.

Reglas de enrutamiento obligatorias:
- Usa compare_official_statistics únicamente cuando la pregunta nombre
  explícitamente dos categorías, sectores, departamentos o períodos a comparar.
  Nunca uses lookup_official_statistic como primera tool para una comparación.
- Para preguntas de identificación o ranking sin dos categorías explícitas, como
  "cuál fue la principal razón", "qué sector tuvo la mayor proporción", "cuál
  fue el menor valor" o "qué categoría lideró", usa
  search_dane_knowledge_base una sola vez. No intentes comparaciones por pares.
- Si la pregunta solicita una cifra puntual, llama primero a
  lookup_official_statistic. Incluye row_label, year y unit cuando estén en la
  pregunta. Usa unit="unidades" para conteos y unit="porcentaje" para
  proporciones.
- Usa search_dane_knowledge_base directamente para definiciones, metodología,
  contexto y explicaciones documentales.

Reglas de parada obligatorias:
- Si lookup_official_statistic devuelve found=true, usa esos datos y responde
  de inmediato. NO llames search_dane_knowledge_base.
- Si lookup_official_statistic devuelve found=false o ambiguous=true, puedes
  llamar search_dane_knowledge_base una sola vez antes de responder.
- Si compare_official_statistics devuelve pares comparables, usa esos datos y
  responde de inmediato. NO llames ninguna otra tool.
- Si compare_official_statistics no devuelve pares, responde exactamente:
  "No tengo evidencia suficiente todavía para responder la pregunta."
  NO intentes construir la comparación con llamadas individuales a
  lookup_official_statistic.

Nunca inventes cifras. Para datos tabulares, informa valor, unidad, año, cuadro,
hoja y URL fuente. Si los resultados son ambiguos, pide una aclaración. Si ninguna
tool aporta evidencia suficiente, responde exactamente: "No tengo evidencia
suficiente todavía para responder la pregunta."
""".strip()


def fts_query(query: str) -> str:
    tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    return " OR ".join(tokens[:12])


def normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    return " ".join(re.findall(r"\w+", decomposed, flags=re.UNICODE))


def row_label_match_score(requested: str, candidate: str) -> float:
    requested_normalized = normalized_text(requested)
    candidate_normalized = normalized_text(candidate)
    if not requested_normalized or not candidate_normalized:
        return 0.0
    if requested_normalized == candidate_normalized:
        return 1.0
    if requested_normalized in candidate_normalized or candidate_normalized in requested_normalized:
        return 0.98

    requested_tokens = set(requested_normalized.split())
    candidate_tokens = set(candidate_normalized.split())
    token_coverage = len(requested_tokens & candidate_tokens) / len(requested_tokens)
    similarity = SequenceMatcher(None, requested_normalized, candidate_normalized).ratio()
    return max(token_coverage, similarity)


class DaneTools:
    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        self.database_path = database_path

    @staticmethod
    def _fetch_statistics_rows(
        connection: sqlite3.Connection,
        match: str,
        row_label: str,
        year: str,
        unit: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT s.*, -bm25(statistics_fts) AS relevance
            FROM statistics_fts
            JOIN statistics AS s ON s.id = statistics_fts.rowid
            WHERE statistics_fts MATCH ?
              AND (? = '' OR lower(s.row_label) LIKE '%' || lower(?) || '%')
              AND (? = '' OR s.year = ?)
              AND (? = '' OR lower(s.unit) = lower(?))
            ORDER BY relevance DESC
            LIMIT ?
            """,
            (match, row_label, row_label, year, year, unit, unit, limit),
        ).fetchall()

    @staticmethod
    def _best_matching_row_label(
        connection: sqlite3.Connection,
        match: str,
        requested_label: str,
        year: str,
        unit: str,
    ) -> str | None:
        candidates = connection.execute(
            """
            SELECT DISTINCT s.row_label
            FROM statistics_fts
            JOIN statistics AS s ON s.id = statistics_fts.rowid
            WHERE statistics_fts MATCH ?
              AND (? = '' OR s.year = ?)
              AND (? = '' OR lower(s.unit) = lower(?))
            LIMIT 500
            """,
            (match, year, year, unit, unit),
        ).fetchall()
        candidates_by_normalized_label = {}
        for candidate in candidates:
            label = candidate["row_label"]
            normalized_label = normalized_text(label)
            current = candidates_by_normalized_label.get(normalized_label)
            if current is None or len(label) < len(current):
                candidates_by_normalized_label[normalized_label] = label

        scored = sorted(
            (
                (row_label_match_score(requested_label, label), label)
                for label in candidates_by_normalized_label.values()
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < ROW_LABEL_MATCH_THRESHOLD:
            return None
        if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.05:
            return None
        return scored[0][1]

    @staticmethod
    def _select_metric_rows(rows: list[dict]) -> tuple[list[dict], bool]:
        best_by_metric = {}
        for row in rows:
            key = (row["table_key"], row["metric"], row["unit"], row["year"])
            if key not in best_by_metric or row["relevance"] > best_by_metric[key]["relevance"]:
                best_by_metric[key] = row

        ranked = sorted(
            best_by_metric.items(), key=lambda item: item[1]["relevance"], reverse=True
        )
        if not ranked:
            return [], False
        if len(ranked) > 1:
            best_score = ranked[0][1]["relevance"]
            second_score = ranked[1][1]["relevance"]
            if second_score >= best_score * RELEVANCE_DOMINANCE_RATIO:
                return rows, True

        selected_key = ranked[0][0]
        return [row for row in rows if (row["table_key"], row["metric"], row["unit"], row["year"]) == selected_key], False

    def search_dane_knowledge_base(self, query: str) -> dict:
        """Busca documentos DANE con recuperación híbrida E5 y BM25.

        Úsala para conceptos, definiciones, metodología, alcance de encuestas,
        explicaciones de indicadores y contexto de publicaciones.

        Args:
            query: Pregunta o términos de búsqueda del usuario.
        """
        results = hybrid_search(query)
        return {
            "query": query,
            "results": [
                {
                    "score_hybrid": round(result["score"], 4),
                    "score_dense": round(result["dense_score"], 4),
                    "score_bm25": round(result["bm25_score"], 4),
                    "text": result["text"][:1600],
                }
                for result in results
            ],
        }

    def lookup_official_statistic(
        self,
        query: str,
        row_label: str = "",
        year: str = "",
        unit: str = "",
        limit: int = DEFAULT_LIMIT,
    ) -> dict:
        """Busca cifras oficiales en los anexos Excel normalizados.

        Úsala para valores, porcentajes, montos, conteos o indicadores. No la
        uses para comparar dos categorías: usa compare_official_statistics.

        Args:
            query: Términos del indicador, cuadro o métrica que se busca.
            row_label: Sector, departamento, categoría o fila a filtrar.
            year: Año de referencia opcional.
            unit: "unidades" para conteos o "porcentaje" para proporciones.
            limit: Máximo de cifras candidatas a devolver.
        """
        match = fts_query(query)
        if not match:
            return {
                "found": False,
                "ambiguous": False,
                "reason": "La consulta no contiene términos buscables.",
                "recommended_next_tool": "search_dane_knowledge_base",
                "results": [],
            }

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        requested_row_label = row_label
        matched_row_label = row_label
        try:
            result_limit = min(max(limit, 1), 20)
            rows = self._fetch_statistics_rows(
                connection, match, row_label, year, unit, result_limit
            )
            if row_label and not rows:
                matched_row_label = self._best_matching_row_label(
                    connection, match, row_label, year, unit
                )
                if matched_row_label:
                    rows = self._fetch_statistics_rows(
                        connection, match, matched_row_label, year, unit, result_limit
                    )
        finally:
            connection.close()

        results = [dict(row) for row in rows]
        results, ambiguous = self._select_metric_rows(results)
        if not results:
            reason = "No se encontraron cifras con los filtros solicitados."
        elif ambiguous:
            reason = "Hay varias métricas candidatas; especifica una unidad o una métrica más precisa."
        else:
            reason = ""
        return {
            "found": bool(results) and not ambiguous,
            "ambiguous": ambiguous,
            "reason": reason,
            "recommended_next_tool": "search_dane_knowledge_base" if reason else None,
            "query": query,
            "row_label_filter": requested_row_label,
            "matched_row_label": matched_row_label,
            "year_filter": year,
            "unit_filter": unit,
            "results": results,
        }

    def compare_official_statistics(
        self,
        metric_query: str,
        row_label_a: str,
        row_label_b: str,
        year: str = "",
        unit: str = "",
    ) -> dict:
        """Compara una misma métrica oficial entre dos filas o categorías.

        Úsala para comparar sectores, departamentos, áreas o períodos presentes
        en un mismo cuadro de los anexos normalizados.

        Args:
            metric_query: Términos de la métrica o indicador a comparar.
            row_label_a: Primera categoría, sector o departamento.
            row_label_b: Segunda categoría, sector o departamento.
            year: Año opcional para restringir la comparación.
            unit: Unidad de la métrica, por ejemplo "unidades" o "porcentaje".
        """
        first = self.lookup_official_statistic(
            query=metric_query,
            row_label=row_label_a,
            year=year,
            unit=unit,
            limit=20,
        )
        second = self.lookup_official_statistic(
            query=metric_query,
            row_label=row_label_b,
            year=year,
            unit=unit,
            limit=20,
        )

        if not first["found"] or not second["found"]:
            return {
                "found": False,
                "ambiguous": first.get("ambiguous") or second.get("ambiguous"),
                "message": "No se encontraron las dos cifras requeridas.",
                "recommended_next_tool": "search_dane_knowledge_base",
                "first_candidates": first.get("results", []),
                "second_candidates": second.get("results", []),
            }

        matching_pairs = [
            (left, right)
            for left in first["results"]
            for right in second["results"]
            if left["table_key"] == right["table_key"]
            and left["metric"] == right["metric"]
            and left["unit"] == right["unit"]
        ]
        if not matching_pairs:
            return {
                "found": False,
                "ambiguous": True,
                "message": "Las cifras encontradas no pertenecen a la misma métrica o cuadro.",
                "recommended_next_tool": "search_dane_knowledge_base",
                "first_candidates": first["results"],
                "second_candidates": second["results"],
            }

        left, right = matching_pairs[0]
        difference = right["value"] - left["value"]
        return {
            "found": True,
            "ambiguous": False,
            "metric": left["metric"],
            "unit": left["unit"],
            "table_key": left["table_key"],
            "title": left["title"],
            "year": left["year"],
            "first": left,
            "second": right,
            "absolute_change": difference,
            "relative_change_pct": None if left["value"] == 0 else difference / abs(left["value"]) * 100,
            "answer": (
                f"{left['row_label']}: {left['value']} {left['unit']}; "
                f"{right['row_label']}: {right['value']} {right['unit']}. "
                f"La diferencia (segunda menos primera) es {difference} {left['unit']}."
            ),
        }


def build_runner():
    load_dotenv()
    tools = Tools()
    tools.add_tools(DaneTools())
    return OpenAIResponsesRunner(
        tools=tools,
        developer_prompt=SYSTEM_PROMPT,
        chat_interface=IPythonChatInterface(),
        llm_client=OpenAIClient(
            model=AGENT_MODEL,
            client=OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=0),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("search", "lookup", "compare"))
    parser.add_argument("--query", default="")
    parser.add_argument("--row-label", default="")
    parser.add_argument("--row-label-b", default="")
    parser.add_argument("--year", default="")
    args = parser.parse_args()
    toolset = DaneTools()

    if args.tool == "search":
        print(json.dumps(toolset.search_dane_knowledge_base(args.query), ensure_ascii=False, indent=2))
        return
    if args.tool == "lookup":
        print(json.dumps(toolset.lookup_official_statistic(args.query, args.row_label, args.year), ensure_ascii=False, indent=2))
        return
    if args.tool == "compare":
        print(json.dumps(toolset.compare_official_statistics(args.query, args.row_label, args.row_label_b, args.year), ensure_ascii=False, indent=2))
        return

    build_runner().run()


if __name__ == "__main__":
    main()
