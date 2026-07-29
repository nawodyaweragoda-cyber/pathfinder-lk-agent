"""Central configuration for PathFinder LK.

Model selection strategy (see README table):
- FAST_MODEL   -> cheap/low-latency Groq model for routing, planning decomposition,
                  and retrieval re-ranking (simple, high-frequency decisions).
- STRONG_MODEL -> larger Groq model reserved for final synthesis and reflection,
                  where reasoning quality matters most.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Models ---------------------------------------------------------------
FAST_MODEL = "llama-3.1-8b-instant"        # routing / re-ranking / decomposition
STRONG_MODEL = "llama-3.3-70b-versatile"   # synthesis + self-critique

# --- RAG ------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pathfinder_lk_corpus"
CORPUS_DIR = os.path.join("data", "corpus")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVE = 8   # candidates pulled from the vector store
TOP_K_FINAL = 4      # kept after LLM re-ranking


def get_groq_api_key() -> str:
    """Load the Groq key from env vars (.env locally) or Streamlit secrets (cloud).

    The key must never be hard-coded or committed. See .env.example.
    """
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:  # Streamlit Cloud deployment
        import streamlit as st

        return st.secrets["GROQ_API_KEY"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file (see .env.example) "
            "or add it to Streamlit secrets."
        ) from exc
