"""Planner Agent — PATTERN 2: Planning / task decomposition.

For itinerary requests, breaks the big request into 2-5 focused retrieval
sub-queries (one per destination/topic) so the Retrieval Agent can ground each
part of the trip in the knowledge base separately.
Uses the FAST model: decomposition here is structured extraction, not deep
reasoning, so the cheap model is sufficient.
"""

from config import FAST_MODEL
from llm import chat_json
from protocol import AgentMessage

AGENT_NAME = "planner"

SYSTEM = """You are the planning component of PathFinder LK, a Sri Lanka travel
assistant. Given a trip request, decompose it into 2-5 focused retrieval
sub-queries, each targeting ONE destination or topic that must be researched
in the knowledge base (e.g. "Sigiriya visiting hours and tickets",
"Kandy to Ella scenic train"). Cover every place or need mentioned; if the
user gave no places, pick sensible classic Sri Lanka stops that fit their
duration and interests.

Return JSON: {"sub_queries": ["...", "..."], "trip_summary": "<one sentence>"}"""


def plan(msg: AgentMessage) -> AgentMessage:
    query = msg.content["query"]
    result = chat_json(FAST_MODEL, SYSTEM, query)
    sub_queries = [q for q in result.get("sub_queries", []) if isinstance(q, str)][:5]
    if not sub_queries:
        sub_queries = [query]
    return msg.reply(
        sender=AGENT_NAME,
        performative="inform",
        content={
            "query": query,
            "sub_queries": sub_queries,
            "trip_summary": result.get("trip_summary", ""),
        },
    )
