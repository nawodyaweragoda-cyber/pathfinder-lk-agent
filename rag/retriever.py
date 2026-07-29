"""Query helper for the persistent Chroma vector store."""

from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_RETRIEVE


@lru_cache(maxsize=1)
def _collection():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        # Cloud cold start: chroma_db/ is gitignored, so build the index now.
        from rag.ingest import main as ingest_main

        ingest_main()
        return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(query: str, k: int = TOP_K_RETRIEVE) -> list[dict]:
    """Return the top-k chunks as [{'text', 'source', 'distance'}, ...]."""
    res = _collection().query(query_texts=[query], n_results=k)
    out = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({"text": text, "source": meta.get("source", "?"), "distance": dist})
    return out
