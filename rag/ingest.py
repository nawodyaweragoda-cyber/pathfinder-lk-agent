"""Ingest the domain corpus into a persistent Chroma vector store.

Chunking strategy: fixed-size sliding window (CHUNK_SIZE chars with
CHUNK_OVERLAP overlap) applied per document, after splitting on blank lines so
chunks rarely cut a sentence mid-way. Each chunk keeps its source filename as
metadata so the synthesis agent can cite sources.

Run once (and re-run whenever data/corpus changes):
    python -m rag.ingest
"""

import glob
import os
import sys

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    CORPUS_DIR,
    EMBEDDING_MODEL,
)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= size:
            buffer = f"{buffer}\n\n{para}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            # carry a tail of the previous chunk forward as overlap
            tail = buffer[-overlap:] if buffer else ""
            buffer = f"{tail}\n\n{para}".strip()[: size * 2]
    if buffer:
        chunks.append(buffer)
    return chunks


def main() -> None:
    files = sorted(
        glob.glob(os.path.join(CORPUS_DIR, "*.md"))
        + glob.glob(os.path.join(CORPUS_DIR, "*.txt"))
    )
    if not files:
        raise SystemExit(f"No corpus files found in {CORPUS_DIR}/")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild from scratch so deletions in the corpus are reflected too.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)

    ids, docs, metas = [], [], []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        source = os.path.basename(path)
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}::{i}")
            docs.append(chunk)
            metas.append({"source": source, "chunk": i})

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested {len(files)} documents -> {len(docs)} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
