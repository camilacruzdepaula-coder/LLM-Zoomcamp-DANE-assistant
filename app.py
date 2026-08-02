"""Interfaz Streamlit y dashboard de observabilidad para el agente DANE."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dane_assistant.monitoring.judge import DaneRelevanceJudge, JUDGE_BATCH_SIZE
from dane_assistant.monitoring.store import ObservabilityStore
from dane_assistant.services.agent_service import DaneAgentService


st.set_page_config(page_title="DANE Stats Assistant", page_icon="D", layout="wide")


@st.cache_resource
def get_agent_service() -> DaneAgentService:
    return DaneAgentService()


def get_observability_store() -> ObservabilityStore:
    return ObservabilityStore()


@st.cache_resource
def get_relevance_judge() -> DaneRelevanceJudge:
    return DaneRelevanceJudge()


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_messages", None)


def display_run_details(record) -> None:
    tool_names = [call["name"] for call in record.tool_calls]
    with st.expander("Detalles de ejecución"):
        metric_columns = st.columns(4)
        metric_columns[0].metric("Modelo", record.model)
        metric_columns[1].metric("Latencia", f"{record.response_time_seconds:.2f} s")
        metric_columns[2].metric("Tokens", record.total_tokens)
        metric_columns[3].metric("Coste", f"${record.total_cost:.5f}")
        st.caption("Tools: " + (", ".join(tool_names) if tool_names else "Ninguna"))


def feedback_controls(store: ObservabilityStore, interaction_id: str) -> None:
    columns = st.columns([1, 1, 6])
    if columns[0].button("Útil", key=f"positive-{interaction_id}"):
        store.save_feedback(interaction_id, 1)
        st.toast("Feedback guardado")
    if columns[1].button("No útil", key=f"negative-{interaction_id}"):
        store.save_feedback(interaction_id, -1)
        st.toast("Feedback guardado")


def evaluate_pending_interactions(store: ObservabilityStore) -> str | None:
    pending = store.pending_llm_judge_interactions(JUDGE_BATCH_SIZE)
    if len(pending) < JUDGE_BATCH_SIZE:
        return None

    interaction_ids = [interaction["interaction_id"] for interaction in pending]
    try:
        result = get_relevance_judge().evaluate(pending)
    except Exception as error:
        store.save_llm_judge_error(
            interaction_ids,
            f"{type(error).__name__}: {error}",
        )
        return "error"

    store.save_llm_judgements(
        [
            {
                "interaction_id": assessment.interaction_id,
                "score": assessment.score,
                "reason": assessment.reason,
            }
            for assessment in result.assessments
        ],
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        total_cost=result.total_cost,
        response_time_seconds=result.response_time_seconds,
    )
    return "evaluated"


def render_chat(store: ObservabilityStore) -> None:
    st.title("Asistente de estadísticas oficiales del DANE")
    st.caption(
        "Consulta indicadores demográficos, sociales, económicos y sectoriales de "
        "Colombia a partir de fuentes oficiales del DANE."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                display_run_details(message["record"])
                feedback_controls(store, message["record"].interaction_id)

    question = st.chat_input("Pregunta aquí")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando fuentes oficiales..."):
            result = get_agent_service().ask(
                question,
                st.session_state.conversation_messages,
            )
            store.save_interaction(result.record)

        judge_status = evaluate_pending_interactions(store)

        if result.record.error:
            st.error("No fue posible procesar la pregunta. Intenta nuevamente.")
            answer = "No fue posible generar una respuesta en este momento."
        else:
            answer = result.record.answer
            st.markdown(answer)
            st.session_state.conversation_messages = result.conversation_messages

        display_run_details(result.record)
        feedback_controls(store, result.record.interaction_id)
        if judge_status == "evaluated":
            st.toast("Se evaluaron tres respuestas con el LLM judge")
        elif judge_status == "error":
            st.warning("No fue posible completar la evaluación automática del lote.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "record": result.record}
    )


def render_dashboard(store: ObservabilityStore) -> None:
    st.title("Monitoring Dashboard")
    summary = store.summary()
    metric_columns = st.columns(6)
    metric_columns[0].metric("Interacciones", summary["interaction_count"])
    metric_columns[1].metric("Latencia media", f"{summary['average_latency']:.2f} s")
    total_cost = summary["total_cost"] + summary["llm_judge_cost"]
    metric_columns[2].metric("Coste total", f"${total_cost:.5f}")
    metric_columns[3].metric("Coste LLM judge", f"${summary['llm_judge_cost']:.5f}")
    user_approval = (
        "Sin feedback"
        if summary["approval_rate"] is None
        else f"{summary['approval_rate']:.0%}"
    )
    judge_relevance = (
        "Sin evaluar"
        if summary["llm_relevance_rate"] is None
        else f"{summary['llm_relevance_rate']:.0%}"
    )
    metric_columns[4].metric("Feedback usuario", user_approval)
    metric_columns[5].metric("LLM judge relevante", judge_relevance)

    daily = pd.DataFrame(store.daily_metrics())
    tools = pd.DataFrame(store.tool_usage())
    feedback = pd.DataFrame(store.feedback_breakdown())
    judge_feedback = pd.DataFrame(store.llm_judge_breakdown())

    if daily.empty:
        st.info("Aún no hay interacciones registradas. Usa el chat para generar métricas.")
        return

    first_row = st.columns(2)
    with first_row[0]:
        st.subheader("Interactions by Day")
        st.line_chart(daily.set_index("day")["interactions"])
    with first_row[1]:
        st.subheader("Cost by Day")
        st.line_chart(daily.set_index("day")["total_cost"])

    second_row = st.columns(2)
    with second_row[0]:
        st.subheader("Average Latency by Day")
        st.line_chart(daily.set_index("day")["average_latency"])
    with second_row[1]:
        st.subheader("Total Tokens by Day")
        st.line_chart(daily.set_index("day")["total_tokens"])

    third_row = st.columns(2)
    with third_row[0]:
        st.subheader("Tool Usage")
        if tools.empty:
            st.caption("No hubo llamadas a tools.")
        else:
            st.bar_chart(tools.set_index("tool")["calls"])
    with third_row[1]:
        st.subheader("User Feedback")
        st.bar_chart(feedback.set_index("feedback")["count"])

    st.subheader("LLM Judge")
    judge_columns = st.columns(3)
    judge_columns[0].metric("Interacciones juzgadas", summary["llm_judged_count"])
    judge_columns[1].metric("Tokens del judge", summary["llm_judge_total_tokens"])
    judge_columns[2].metric("Tokens de entrada/salida", f"{summary['llm_judge_input_tokens']} / {summary['llm_judge_output_tokens']}")
    st.bar_chart(judge_feedback.set_index("feedback")["count"])

    with st.expander("Últimas interacciones"):
        interactions = pd.DataFrame(store.list_interactions(limit=20))
        if not interactions.empty:
            st.dataframe(
                interactions[
                    [
                        "timestamp",
                        "question",
                        "answer",
                        "model",
                        "tool_count",
                        "response_time_seconds",
                        "total_cost",
                        "feedback",
                        "llm_judge_score",
                        "llm_judge_cost",
                        "error",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )


def main() -> None:
    initialize_session()
    store = get_observability_store()
    chat_tab, dashboard_tab = st.tabs(["Chat", "Monitoring Dashboard"])
    with chat_tab:
        render_chat(store)
    with dashboard_tab:
        render_dashboard(store)


if __name__ == "__main__":
    main()
