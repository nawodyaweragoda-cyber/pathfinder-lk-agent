"""Synthesis Agent — PATTERN 4: Reflection / self-critique.

Drafts the final answer from retrieved context using the STRONG model, then
runs a critique pass over its own draft (grounding, completeness, hallucination
check) and revises once. The propose -> critique -> final message sequence is
visible in the trace, which demonstrates the reflection pattern explicitly.
"""

from config import STRONG_MODEL
from llm import chat
from protocol import AgentMessage

AGENT_NAME = "synthesis"

DRAFT_SYSTEM = """You are PathFinder LK, a friendly Sri Lanka travel expert.
Answer the user's question or build their itinerary using ONLY the provided
context chunks. Cite the source file in brackets after facts, e.g. [ella.md].
If the context is missing something important, say so honestly instead of
inventing details. Keep the tone warm and practical."""

CRITIQUE_SYSTEM = """You are a strict reviewer. Given a user query, source
context, and a draft answer, list concrete problems: claims not supported by
the context, missing information the context could have answered, unclear
structure. If the draft is good, say "APPROVED". Be brief (max 5 bullet
points)."""

REVISE_SYSTEM = """You are PathFinder LK. Improve the draft answer using the
reviewer's critique. Stay grounded in the context only. Return just the final
answer."""


def _format_context(context: list[dict]) -> str:
    parts = []
    for item in context:
        for chunk in item["chunks"]:
            parts.append(f"[{chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts) if parts else "(no context retrieved)"


def synthesize(msg: AgentMessage, trace_log: list[AgentMessage]) -> AgentMessage:
    query = msg.content["query"]
    ctx = _format_context(msg.content.get("context", []))

    draft = chat(
        STRONG_MODEL, DRAFT_SYSTEM, f"Context:\n{ctx}\n\nUser query: {query}", 0.4
    )
    proposal = msg.reply(
        sender=AGENT_NAME, performative="propose", content={"draft": draft}
    )
    trace_log.append(proposal)

    critique = chat(
        STRONG_MODEL,
        CRITIQUE_SYSTEM,
        f"User query: {query}\n\nContext:\n{ctx}\n\nDraft answer:\n{draft}",
        0.2,
    )
    critique_msg = proposal.reply(
        sender=AGENT_NAME, performative="critique", content={"critique": critique}
    )
    trace_log.append(critique_msg)

    if "APPROVED" in critique.upper():
        final = draft
    else:
        final = chat(
            STRONG_MODEL,
            REVISE_SYSTEM,
            f"User query: {query}\n\nContext:\n{ctx}\n\n"
            f"Draft:\n{draft}\n\nCritique:\n{critique}",
            0.4,
        )

    return msg.reply(
        sender=AGENT_NAME,
        performative="final",
        content={"answer": final, "critique": critique},
    )
