import json
import math
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dane_assistant.config import RAG_INDEX_DIR


# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------

INDEX_DIR = RAG_INDEX_DIR
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
BM25_PATH = INDEX_DIR / "bm25_index.json"

# Debe ser el mismo modelo con el que creaste los embeddings.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Modelo de OpenAI que generará la respuesta final.
LLM_MODEL = "gpt-4o-mini"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.70
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5


# -------------------------------------------------------------------
# Cargar recursos
# -------------------------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "No se encontró OPENAI_API_KEY. "
        "Crea un archivo .env con: OPENAI_API_KEY=tu_clave"
    )

print("Cargando modelo de embeddings...")
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    local_files_only=True,
)

print("Cargando chunks...")
with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

print("Cargando embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Cargando índice BM25...")
with BM25_PATH.open("r", encoding="utf-8") as file:
    bm25_index = json.load(file)

if len(chunks) != len(embeddings):
    raise ValueError(
        f"La cantidad de chunks ({len(chunks)}) no coincide con "
        f"la cantidad de embeddings ({len(embeddings)})."
    )

if bm25_index["document_count"] != len(chunks):
    raise ValueError(
        "No coinciden la cantidad de chunks y documentos del índice BM25. "
        "Run the ingestion pipeline to rebuild the index."
    )

if not math.isclose(DENSE_WEIGHT + BM25_WEIGHT, 1.0):
    raise ValueError("DENSE_WEIGHT y BM25_WEIGHT deben sumar 1.0.")

# Los embeddings se generaron normalizados en embed_chunks.py.
# Por tanto, el producto punto con la consulta normalizada es similitud coseno.


# -------------------------------------------------------------------
# Funciones del RAG
# -------------------------------------------------------------------

def get_chunk_text(chunk):
    """
    Obtiene el texto de un chunk.

    Cambia esta función únicamente si tus chunks usan otro nombre
    para guardar el texto, por ejemplo: 'content' o 'chunk'.
    """
    if isinstance(chunk, str):
        return chunk

    return chunk["text"]


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def bm25_scores(question):
    """Calcula el score BM25 de la pregunta contra todos los chunks."""
    scores = np.zeros(bm25_index["document_count"], dtype=np.float32)
    document_count = bm25_index["document_count"]
    average_length = bm25_index["average_document_length"]
    document_lengths = bm25_index["document_lengths"]
    postings = bm25_index["postings"]
    k1 = bm25_index["k1"]
    b = bm25_index["b"]

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


def normalize_scores(scores):
    """Lleva scores de una consulta al rango de 0 a 1 para combinarlos."""
    minimum = scores.min()
    maximum = scores.max()

    if math.isclose(float(minimum), float(maximum)):
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def search(question):
    """
    Genera el embedding de la pregunta, toma los cinco resultados con mayor
    similitud y conserva los que superan el umbral configurado.
    """

    # E5 recomienda usar el prefijo 'query:' para las consultas.
    question_embedding = embedding_model.encode(
        f"query: {question}",
        normalize_embeddings=True,
    )

    # Como los embeddings están normalizados, este producto punto
    # representa la similitud coseno.
    dense_scores = embeddings @ question_embedding
    lexical_scores = bm25_scores(question)

    dense_normalized = normalize_scores(dense_scores)
    lexical_normalized = normalize_scores(lexical_scores)
    dense_contributions = DENSE_WEIGHT * dense_normalized
    bm25_contributions = BM25_WEIGHT * lexical_normalized
    hybrid_scores = dense_contributions + bm25_contributions

    # Primero tomamos los cinco resultados con mayor score híbrido.
    top_indices = np.argsort(hybrid_scores)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        if hybrid_scores[index] <= SIMILARITY_THRESHOLD:
            break

        results.append(
            {
                "text": get_chunk_text(chunks[index]),
                "score": float(hybrid_scores[index]),
                "dense_score": float(dense_scores[index]),
                "bm25_score": float(lexical_scores[index]),
                "dense_contribution": float(dense_contributions[index]),
                "bm25_contribution": float(bm25_contributions[index]),
            }
        )

    return results


def build_prompt(question, results):
    """
    Une los chunks recuperados y los coloca en el prompt.
    """

    context = "\n\n---\n\n".join(
        f"CHUNK {position}:\n{result['text']}"
        for position, result in enumerate(results, start=1)
    )

    prompt = f"""
Eres un asistente virtual del DANE (Departamento Administrativo Nacional de Estadística de Colombia).

Tu función es responder preguntas generales sobre estadísticas, operaciones estadísticas,
procesos, metodologías, trámites, documentos y demás información institucional del DANE,
utilizando únicamente la información recuperada de los documentos disponibles.

Responde la PREGUNTA de forma clara, precisa y en español.

Reglas importantes:
- Usa exclusivamente la información incluida en el CONTEXTO.
- No inventes datos, cifras, fechas, enlaces, requisitos ni procedimientos.
- No uses conocimiento externo, aunque conozcas la respuesta.
- Si el contexto contiene información parcial, responde únicamente lo que esté respaldado
  por el contexto y aclara qué información no está disponible.
- Si el contexto no contiene información suficiente para responder, di exactamente:
  "No encontré información suficiente en los documentos."

PREGUNTA:
{question}

CONTEXTO:
{context}
""".strip()

    return prompt


def ask_llm(prompt):
    """
    Envía el prompt a OpenAI y devuelve la respuesta.
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def rag(question):
    """
    Flujo completo:
    pregunta -> top 5 por similitud -> filtro por umbral -> prompt -> LLM.
    """

    results = search(question)

    if not results:
        return "No encontré información suficiente en los documentos.", results

    prompt = build_prompt(question, results)
    answer = ask_llm(prompt)

    return answer, results


# -------------------------------------------------------------------
# Chat en consola
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("\nRAG listo. Escribe una pregunta.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        question = input("Pregunta: ").strip()

        if question.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        if not question:
            continue

        answer, retrieved_chunks = rag(question)

        print("\nChunks recuperados:")
        for position, result in enumerate(retrieved_chunks, start=1):
            print(
                f"\n--- Chunk {position} | híbrido: {result['score']:.4f} | "
                f"denso: {result['dense_score']:.4f} | "
                f"BM25: {result['bm25_score']:.4f} ---"
            )
            print(
                "Aportes: "
                f"denso {result['dense_contribution']:.4f} "
                f"({result['dense_contribution'] / result['score']:.1%}) | "
                f"BM25 {result['bm25_contribution']:.4f} "
                f"({result['bm25_contribution'] / result['score']:.1%})"
            )
            print(result["text"][:500])

        print("\nRespuesta:")
        print(answer)
        print()
