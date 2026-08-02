import argparse
import json
from pathlib import Path

# Configuración inicial.
# Más adelante puedes comparar estos valores en la evaluación.
CHUNK_SIZE = 1000
OVERLAP = 100

DEFAULT_INPUT_PATH = Path("build/dane-ingestion/sources/documents/processed/documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("build/dane-ingestion/sources/documents/processed/chunks.jsonl")


def make_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Divide texto en bloques de caracteres con overlap.

    Ejemplo:
    - chunk_size = 1000
    - overlap = 100
    - cada nuevo chunk empieza 900 caracteres después del anterior.
    """
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def build_text_for_embedding(document, content):
    """
    Equivalente a la idea del curso de unir campos para crear el texto
    que será convertido en embedding.
    """
    location = document.get("location", {})

    return f"""
Tema: {document.get("topic", "")}
Subtema: {document.get("subtopic", "")}
Publicación: {document.get("publication_title", "")}
Tipo de documento: {document.get("document_type", "")}
Periodo de referencia: {document.get("reference_period", "")}
Ubicación: página {location.get("page", "")}, hoja {location.get("sheet", "")}

Contenido:
{content}
""".strip()


def build_chunks(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_documents = 0
    total_chunks = 0
    tabular_documents_skipped = 0

    with input_path.open("r", encoding="utf-8") as input_file, \
         output_path.open("w", encoding="utf-8") as output_file:

        for line in input_file:
            line = line.strip()

            if not line:
                continue

            document = json.loads(line)
            total_documents += 1

            if document.get("extraction_method") in {"excel", "csv"}:
                tabular_documents_skipped += 1
                continue

            text = document.get("text", "").strip()

            if not text:
                continue

            chunks = make_chunks(text)

            for index, content in enumerate(chunks):
                chunk = {
                    "chunk_id": f"{document['document_id']}_chunk_{index}",
                    "document_id": document["document_id"],
                    "chunk_index": index,

                    # Este será el texto que enviaremos al modelo de embeddings.
                    "text": build_text_for_embedding(document, content),

                    # Conservamos el contenido sin metadata para mostrarlo luego.
                    "content": content,

                    # Metadata para filtros y citas.
                    "topic": document.get("topic"),
                    "subtopic": document.get("subtopic"),
                    "publication_title": document.get("publication_title"),
                    "document_type": document.get("document_type"),
                    "reference_period": document.get("reference_period"),
                    "source_url": document.get("source_url"),
                    "landing_url": document.get("landing_url"),
                    "asset_filename": document.get("asset_filename"),
                    "location": document.get("location"),
                }

                output_file.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

                total_chunks += 1

    print(f"Documentos procesados: {total_documents}")
    print(f"Documentos tabulares excluidos: {tabular_documents_skipped}")
    print(f"Chunks creados: {total_chunks}")
    print(f"Archivo generado: {output_path}")
    return total_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    build_chunks(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
