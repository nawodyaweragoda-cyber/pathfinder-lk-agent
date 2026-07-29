"""Retrieval Agent — PATTERN 3: Tool use.

Treats the Chroma vector store as an external tool: it retrieves candidate
chunks, then uses the FAST model to re-rank them by actual relevance to the
question (embedding similarity alone often surfaces near-misses). Only the
best chunks are forwarded to the Synthesis Agent.
"""

from config import FAST_MODEL, TOP_K_FINAL
from llm import chat_json
from protocol import AgentMessage
from rag.retriever import retrieve

AGENT_NAME = "retrieval"

RERANK_SYSTEM = """You are a retrieval re-ranker. You get a question and a
numbered list of text chunks. Pick the chunks that genuinely help answer the
question, best first.

Return JSON: {"ranked_indices": [<int>, ...]} using the given chunk numbers."""


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    numbered = "\n\n".join(
        f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(candidates)
    )
    try:
        result = chat_json(
            FAST_MODEL, RERANK_SYSTEM, f"Question: {query}\n\nChunks:\n{numbered}"
        )
        order = [i for i in result.get("ranked_indices", []) if 0 <= i < len(candidates)]
    except Exception:  # noqa: BLE001 — fall back to embedding order
        order = list(range(len(candidates)))
    if not order:
        order = list(range(len(candidates)))
    return [candidates[i] for i in order[:TOP_K_FINAL]]


def fetch(msg: AgentMessage) -> AgentMessage:
    """Handle a 'request' for context on one or more sub-queries."""
    sub_queries = msg.content.get("sub_queries") or [msg.content["query"]]
    context: list[dict] = []
    for sq in sub_queries:
        candidates = retrieve(sq)
        best = _rerank(sq, candidates)
        context.append({"sub_query": sq, "chunks": best})
    return msg.reply(
        sender=AGENT_NAME,
        performative="inform",
        content={"query": msg.content["query"], "context": context},
    )
