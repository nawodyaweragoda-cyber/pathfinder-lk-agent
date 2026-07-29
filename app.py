# PathFinder LK — Streamlit UI.
# Run locally:   streamlit run app.py
# (Ingest the corpus first: python -m rag.ingest)

# Streamlit Cloud ships an old sqlite3; Chroma needs >= 3.35.
# This MUST run before anything imports chromadb (i.e. before orchestrator).
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # local dev on Windows: system sqlite is fine

import streamlit as st

st.set_page_config(page_title="PathFinder LK",
                   page_icon="🧭", layout="centered")

from orchestrator import run  # noqa: E402  (must come after the sqlite shim)


st.title("🧭 PathFinder LK")
st.caption(
    "Multi-agent, RAG-grounded travel assistant for Sri Lanka — "
    "Router → Planner → Retrieval → Synthesis (with self-critique)."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "- **Router** (Llama 3.1 8B) classifies your query\n"
        "- **Planner** (Llama 3.1 8B) decomposes itineraries\n"
        "- **Retrieval** re-ranks chunks from the knowledge base\n"
        "- **Synthesis** (Llama 3.3 70B) drafts, self-critiques, and revises\n"
    )
    st.markdown(
        "Every hop is a structured `AgentMessage` — expand the trace under each answer.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (query, answer, trace)

for query, answer, trace in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("🔍 Agent message trace"):
            for m in trace:
                st.markdown(
                    f"**{m.sender} → {m.receiver}** · `{m.performative}`"
                    + (f" · intent=`{m.intent}`" if m.intent else "")
                )
                st.json(m.content, expanded=False)

prompt = st.chat_input("Ask about Sri Lanka travel, or request an itinerary…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Agents at work…"):
            try:
                answer, trace = run(prompt)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.stop()
        st.markdown(answer)
        with st.expander("🔍 Agent message trace"):
            for m in trace:
                st.markdown(
                    f"**{m.sender} → {m.receiver}** · `{m.performative}`"
                    + (f" · intent=`{m.intent}`" if m.intent else "")
                )
                st.json(m.content, expanded=False)
    st.session_state.history.append((prompt, answer, trace))
