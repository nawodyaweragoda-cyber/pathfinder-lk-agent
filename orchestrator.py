"""Orchestrator — coordinates all agents (orchestrator-worker style wiring).

Flow:
    user -> Router -> (Planner if itinerary) -> Retrieval -> Synthesis -> user

Every hop is an AgentMessage on the same trace_id; the full trace is returned
so the UI can show exactly how agents talked to each other.
"""

from agents import planner_agent, retrieval_agent, router_agent, synthesis_agent
from protocol import AgentMessage


def run(query: str) -> tuple[str, list[AgentMessage]]:
    """Run the full multi-agent pipeline. Returns (answer, message_trace)."""
    trace: list[AgentMessage] = []

    user_msg = AgentMessage.new_trace(
        sender="user",
        receiver=router_agent.AGENT_NAME,
        performative="request",
        content={"query": query},
    )
    trace.append(user_msg)

    routed = router_agent.route(user_msg)
    trace.append(routed)

    if routed.intent == "out_of_scope":
        answer = (
            "I'm PathFinder LK — I can only help with travel in Sri Lanka "
            "(destinations, itineraries, transport, seasons). "
            "Try asking me something like *'Plan 3 days in the hill country'*."
        )
        reject = routed.reply(
            sender="orchestrator", performative="reject", content={"answer": answer}
        )
        trace.append(reject)
        return answer, trace

    if routed.intent == "itinerary_planning":
        plan_req = AgentMessage(
            trace_id=user_msg.trace_id,
            sender="orchestrator",
            receiver=planner_agent.AGENT_NAME,
            performative="request",
            intent=routed.intent,
            content={"query": query},
        )
        trace.append(plan_req)
        planned = planner_agent.plan(plan_req)
        trace.append(planned)
        sub_queries = planned.content["sub_queries"]
    else:
        sub_queries = [query]

    retrieve_req = AgentMessage(
        trace_id=user_msg.trace_id,
        sender="orchestrator",
        receiver=retrieval_agent.AGENT_NAME,
        performative="request",
        intent=routed.intent,
        content={"query": query, "sub_queries": sub_queries},
    )
    trace.append(retrieve_req)
    retrieved = retrieval_agent.fetch(retrieve_req)
    trace.append(retrieved)

    final = synthesis_agent.synthesize(retrieved, trace)
    trace.append(final)

    return final.content["answer"], trace


if __name__ == "__main__":
    ans, tr = run("Plan a 3 day trip covering Sigiriya and Kandy")
    for m in tr:
        print(m.pretty()[:160])
    print("\n=== ANSWER ===\n", ans)
