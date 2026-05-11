"""
Motor & Cyber Law Advisor 
"""

import sys

# --- RENDER SQLITE3 WORKAROUND FOR CHROMADB ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------------------

import time
import asyncio
from pathlib import Path

import streamlit as st

# Fix event-loop issue on Windows
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

sys.path.insert(0, str(Path(__file__).parent))

from retrievalchain import (
    load_vector_store, build_llm, init_db,
    retrieve_chunks, format_context, generate_answer, log_query,
    learn_from_unanswered,
    DISCLAIMER,
)

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Law Advisor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS ───────────────────────────────────────────────────────
# ── Session state ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


# ── Load resources (once) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    db   = load_vector_store()
    llm  = build_llm()
    conn = init_db()
    return db, llm, conn

resource_ok = False
try:
    placeholder = st.empty()
    with placeholder.container():
        st.write("") # Spacer
        st.markdown("<h1 style='text-align:center;'>Law Advisor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Loading Law Advisor...</p>", unsafe_allow_html=True)
        st.write("") # Spacer

    db, llm, conn = load_resources()
    placeholder.empty()
    resource_ok = True
except Exception as e:
    st.error(f"Failed to load database: {e}")

# ── Title ─────────────────────────────────────────────────────────────
st.markdown("<h2 style='text-align:left; margin-bottom:4px;'>Law Advisor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:left; color:#64748B; font-size:13px; margin-top:0;'>Motor Vehicles Act 2019 · IT Act 2000</p>", unsafe_allow_html=True)

# ── FAQ (only when chat is empty) ─────────────────────────────────────
FAQ = [
    "Fine for drunk driving?",
    "Penalty for no driving licence?",
    "Identity theft under IT Act?",
    "Helmet rules for bikers?",
    "What is Section 66C?",
    "Is hacking a crime in India?",
]

if not st.session_state.messages and resource_ok:
    st.markdown("<p style='color:#64748B; font-size:13px; margin:18px 0 8px;'>Frequently asked:</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, q in enumerate(FAQ):
        if cols[i % 3].button(q, key=f"faq_{i}"):
            st.session_state.pending = q
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("learned"):
            st.caption("Learned & saved to database — will answer instantly next time")
        elif msg.get("source"):
            st.caption(f"Source: {msg['source']} · {msg.get('latency_ms','?')}ms")

# Clear button
if st.session_state.messages:
    if st.button("Clear", key="clear"):
        st.session_state.messages = []
        st.rerun()

# ── Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a legal question...", disabled=not resource_ok)

# Resolve which question to answer
question = None
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None
elif user_input:
    question = user_input

# ── Process question ──────────────────────────────────────────────────
if question and resource_ok:
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            t0      = time.time()
            chunks  = retrieve_chunks(db, question, k=3)
            context = format_context(chunks)
            answer  = generate_answer(llm, context, question)
            log_query(conn, question, answer, chunks, t0)
            ms      = int((time.time() - t0) * 1000)
            found   = DISCLAIMER not in answer
            source  = chunks[0].metadata.get("act_name", "") if chunks else ""

        # ── Self-learning: if question not answered, auto-train & save ─────
        if not found:
            with st.spinner("Not found in database — learning this question now..."):
                learned_answer = learn_from_unanswered(db, llm, question)
            
            st.markdown(learned_answer)
            st.caption("Learned & saved to database — will answer instantly next time")
            st.session_state.messages.append({
                "role": "assistant",
                "content": learned_answer,
                "answer_found": True,
                "learned": True,
                "latency_ms": int((time.time() - t0) * 1000),
            })
        else:
            st.markdown(answer)
            st.caption(f"Source: {source} · {ms}ms")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "answer_found": True,
                "learned": False,
                "source": source,
                "latency_ms": ms,
            })