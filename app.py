"""Meeting Memory: a small Streamlit front end.

    pip install streamlit
    streamlit run app.py          # run from the project folder, with your API key set in this terminal

Two pages:
  Ask                    chat with your meetings; every answer lists the lines it came from
  Decisions & actions    the items found by `python -m extract.extract`, with the quote behind each
"""
from __future__ import annotations

import os

import streamlit as st

from ask.ask import answer_question
from extract.llm import PROVIDERS, ExtractionError, make_llm, tokens_used
from store import db

st.set_page_config(page_title="Meeting Memory", page_icon="💬", layout="wide")

EXAMPLES = [
    "What is the project manager's name?",
    "Why was voice recognition ruled out?",
    "What is the capital of France?",
]


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def source_dicts(answer) -> list[dict]:
    return [
        {"n": n, "meeting": w.meeting_id, "start": mmss(w.start), "end": mmss(w.end),
         "lines": [f"{r['speaker']}: {r['text']}" for r in w.rows]}
        for n, w in answer.sources
    ]


def show_message(msg: dict) -> None:
    st.markdown(msg["text"])
    if msg.get("warning"):
        st.warning(msg["warning"])
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"**[S{s['n']}] {s['meeting']} · {s['start']}–{s['end']}**")
                st.text("\n".join(s["lines"]))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Meeting Memory")
    page = st.radio("Page", ["Ask", "Decisions & actions"], label_visibility="collapsed")
    db_path = st.text_input("Database", os.environ.get("MM_DB", db.DEFAULT_DB))

    st.subheader("Model")
    backends = ["groq", "openrouter", "ollama", "openai"]
    default_backend = os.environ.get("MM_BACKEND", "groq")
    backend = st.selectbox("Backend", backends,
                           index=backends.index(default_backend) if default_backend in backends else 0)
    model = st.text_input("Model name", os.environ.get("MM_MODEL", "qwen/qwen3.8-27b"))
    base_url = st.text_input("Base URL (openai backend only)") if backend == "openai" else None
    effort = st.selectbox("Reasoning effort", ["(default)", "low", "medium", "high"])
    max_tokens = st.number_input("Max output tokens", 500, 8000, 2000, step=500)

    if backend in PROVIDERS:
        env = PROVIDERS[backend][1]
        if os.environ.get(env):
            st.success(f"{env} is set")
        else:
            st.error(f"{env} is not set. Set it in the terminal, then restart this app.")
    if backend == "ollama":
        st.caption("Local model: nothing leaves this computer.")
    else:
        st.caption(f"Questions and the retrieved transcript lines are sent to {backend}.")

    st.subheader("Retrieval")
    expand = st.checkbox("Expand question with synonyms", help="One extra model call per question.")
    per_meeting = st.number_input("Best matches from every meeting", 0, 10, 0,
                                  help="0 = off. Try 3 for questions that span several meetings.")
    max_windows = st.number_input("Max source windows", 2, 12, 6)
    st.caption(f"Tokens used this session: {tokens_used():,}")

conn = db.connect(db_path)
meetings = conn.execute(
    "SELECT m.id, COUNT(s.id) AS n FROM meetings m LEFT JOIN segments s ON s.meeting_id = m.id "
    "GROUP BY m.id ORDER BY m.id").fetchall()

if not meetings:
    st.title("Meeting Memory")
    st.info("No meetings are loaded yet. Run, in the project folder:\n\n"
            "`python -m ingest.download_ami`  then  `python -m ingest.add_meeting ami data/ami ES2008a`")
    st.stop()

# ------------------------------------------------------------------ Ask
if page == "Ask":
    st.title("Ask your meetings")
    st.caption("Loaded: " + ", ".join(f"{m['id']} ({m['n']} segments)" for m in meetings))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, width="stretch"):
            st.session_state.queued = example

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            show_message(msg)

    question = st.chat_input("Ask about your meetings...")
    question = st.session_state.pop("queued", None) or question

    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the meetings..."):
                try:
                    llm = make_llm(backend, model, base_url=base_url or None,
                                   max_tokens=int(max_tokens),
                                   reasoning_effort=None if effort == "(default)" else effort)
                    ans = answer_question(conn, question, llm, expand=expand,
                                          per_meeting=int(per_meeting), max_windows=int(max_windows))
                    reply = {"role": "assistant", "text": ans.text,
                             "sources": source_dicts(ans)}
                    if ans.found and not ans.verified:
                        reply["warning"] = "UNVERIFIED: the model gave no valid citation, so treat this as a guess."
                except ExtractionError as exc:
                    reply = {"role": "assistant", "text": "Sorry, I could not get an answer.",
                             "warning": str(exc)}
            show_message(reply)
        st.session_state.messages.append(reply)

    if st.session_state.messages and st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ Decisions & actions
else:
    st.title("Decisions, actions and questions")
    st.caption("Extracted by `python -m extract.extract`. Each item is tied to a quote from the meeting. "
               "Treat questions with care: many were answered a few lines later.")
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Type", ["all", "decision", "action", "question"])
    meeting = c2.selectbox("Meeting", ["all"] + [m["id"] for m in meetings])
    owner_filter = c3.text_input("Owner contains")

    items = db.list_items(conn, type_=None if kind == "all" else kind,
                          meeting_id=None if meeting == "all" else meeting, status=None)
    rows = [
        {"type": i["type"], "owner": i["owner"] or "", "item": i["text"], "meeting": i["meeting_id"],
         "time": mmss(i["cited_start"]), "speaker": i["cited_speaker"], "quote": i["cited_text"]}
        for i in items
        if owner_filter.lower() in (i["owner"] or "").lower()
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"{len(rows)} items")
    else:
        st.info("No items yet. Run `python -m extract.extract ES2008b --backend groq --model <model>`.")

conn.close()
