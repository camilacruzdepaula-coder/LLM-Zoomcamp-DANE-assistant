import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Configuración
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32

DEFAULT_INPUT_PATH = Path("build/dane-ingestion/sources/documents/processed/chunks.jsonl")
DEFAULT_OUTPUT_DIR = Path("build/dane-ingestion/data/rag")

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def build_bm25_index(chunks):
    """Construye un índice léxico compacto para recuperación híbrida."""
    postings = defaultdict(list)
    document_lengths = []

    for document_index, chunk in enumerate(chunks):
        # Usamos contenido sin metadata repetida para priorizar términos puntuales.
        term_frequencies = Counter(tokenize(chunk["content"]))
        document_lengths.append(sum(term_frequencies.values()))

        for term, frequency in term_frequencies.items():
            postings[term].append([document_index, frequency])

    return {
        "document_count": len(chunks),
        "average_document_length": (
            sum(document_lengths) / len(document_lengths)
            if document_lengths
            else 0
        ),
        "document_lengths": document_lengths,
        "k1": BM25_K1,
        "b": BM25_B,
        "postings": dict(postings),
    }


def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def build_index(
    input_path: Path,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = output_dir / "bm25_index.json"

    print("Cargando chunks...")
    chunks = load_chunks(input_path)

    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    print(f"Chunks cargados: {len(chunks)}")
    print(f"Cargando modelo: {model_name}")

    model = SentenceTransformer(model_name)

    print("Generando embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Matriz de vectores: una fila por chunk.
    np.save(output_dir / "embeddings.npy", embeddings)

    # Guardamos metadata y texto, sin duplicar los vectores dentro del JSONL.
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    bm25_index = build_bm25_index(chunks)
    with bm25_path.open("w", encoding="utf-8") as file:
        json.dump(bm25_index, file, ensure_ascii=False)

    print(f"Embeddings guardados: {output_dir / 'embeddings.npy'}")
    print(f"Metadata guardada: {output_dir / 'chunks.jsonl'}")
    print(f"Índice BM25 guardado: {bm25_path}")
    print(f"Forma de la matriz: {embeddings.shape}")
    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    build_index(
        Path(args.input),
        Path(args.output_dir),
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
