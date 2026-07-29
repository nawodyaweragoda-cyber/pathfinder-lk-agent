"""Router Agent — PATTERN 1: Router.

Classifies the user's query into an intent and routes it to the right
downstream agent. Uses the FAST model because this is a cheap, latency-critical
classification task that does not need deep reasoning.
"""

from config import FAST_MODEL
from llm import chat_json
from protocol import AgentMessage

AGENT_NAME = "router"

INTENTS = ["destination_info", "itinerary_planning", "out_of_scope"]

SYSTEM = """You are the routing component of PathFinder LK, a Sri Lanka travel
assistant. Classify the user's query into exactly one intent:

- "destination_info": factual questions about a place, attraction, ticket,
  transport, season, or activity in Sri Lanka.
- "itinerary_planning": the user wants a multi-stop or multi-day trip plan.
- "out_of_scope": anything not related to travel in Sri Lanka.

Return JSON: {"intent": "<one of the three>", "reason": "<one short sentence>"}"""


def route(msg: AgentMessage) -> AgentMessage:
    """Consume a user 'request' message, emit an 'inform' with the intent."""
    query = msg.content["query"]
    result = chat_json(FAST_MODEL, SYSTEM, query)
    intent = result.get("intent", "out_of_scope")
    if intent not in INTENTS:
        intent = "out_of_scope"
    return msg.reply(
        sender=AGENT_NAME,
        performative="inform",
        intent=intent,
        content={"query": query, "reason": result.get("reason", "")},
    )
