"""
Motor & Cyber Law Advisor — Minimal Chat UI
"""

import sys
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
    page_icon="⚖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS ───────────────────────────────────────────────────────
# ── Session state ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

# ── Minimal CSS ───────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

[data-testid="collapsedControl"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0F1117;
    color: #E2E8F0;
}

.user-bubble {
    background: #1E293B;
    border-left: 3px solid #38BDF8;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #F1F5F9;
}
.bot-bubble {
    background: #162032;
    border-left: 3px solid #A78BFA;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #E2E8F0;
    line-height: 1.7;
}
.learned-bubble {
    background: #0D2B1A;
    border-left: 3px solid #34D399;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #A7F3D0;
    line-height: 1.7;
}

.stButton > button {
    background: #1E293B;
    color: #CBD5E1;
    border: 1px solid #334155;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 12px;
    width: 100%;
    text-align: left;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #38BDF8;
    color: #38BDF8;
    background: #0F2133;
}

.source-line { font-size: 11px; color: #475569; margin-top: 6px; }
.learned-badge { font-size: 11px; color: #34D399; margin-top: 6px; }
</style>
"""

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

[data-testid="collapsedControl"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #F8FAFC;
    color: #0F1117;
}

.user-bubble {
    background: #E0F2FE;
    border-left: 3px solid #0284C7;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #0F1117;
}
.bot-bubble {
    background: #F1F5F9;
    border-left: 3px solid #7C3AED;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #0F1117;
    line-height: 1.7;
}
.learned-bubble {
    background: #ECFDF5;
    border-left: 3px solid #059669;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
    color: #0F1117;
    line-height: 1.7;
}

.stButton > button {
    background: #F8FAFC;
    color: #334155;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 12px;
    width: 100%;
    text-align: left;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #0284C7;
    color: #0284C7;
    background: #E0F2FE;
}

.source-line { font-size: 11px; color: #94A3B8; margin-top: 6px; }
.learned-badge { font-size: 11px; color: #059669; margin-top: 6px; }
</style>
"""

if st.session_state.light_mode:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
else:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

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
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            try:
                st.image("logo.png", use_column_width=True)
            except:
                st.markdown("<h1 style='text-align:center;'>⚖</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center;'>Loading Law Advisor...</p>", unsafe_allow_html=True)
        st.write("") # Spacer

    db, llm, conn = load_resources()
    placeholder.empty()
    resource_ok = True
except Exception as e:
    st.error(f"Failed to load database: {e}")

# ── Title ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.markdown("<h2 style='text-align:left; margin-bottom:4px;'>⚖ Law Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:left; color:#64748B; font-size:13px; margin-top:0;'>Motor Vehicles Act 2019 · IT Act 2000</p>", unsafe_allow_html=True)
with col2:
    st.toggle("☀️ Mode", key="light_mode")

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
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        is_learned = msg.get("learned", False)
        css  = "learned-bubble" if is_learned else "bot-bubble"
        icon = "🧠" if is_learned else "⚖"
        st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
        if is_learned:
            st.markdown('<div class="learned-badge">✓ Learned &amp; saved to database — will answer instantly next time</div>', unsafe_allow_html=True)
        elif msg.get("source"):
            st.markdown(f'<div class="source-line">📋 {msg["source"]} · {msg.get("latency_ms","?")}ms</div>', unsafe_allow_html=True)

# Clear button
if st.session_state.messages:
    if st.button("🗑 Clear", key="clear"):
        st.session_state.messages = []
        st.rerun()

# ── Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a legal question…", disabled=not resource_ok)

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

    with st.spinner("Searching…"):
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
        with st.spinner("🧠 Not found in database — learning this question now…"):
            learned_answer = learn_from_unanswered(db, llm, question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": learned_answer,
            "answer_found": True,
            "learned": True,
            "latency_ms": int((time.time() - t0) * 1000),
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "answer_found": True,
            "learned": False,
            "source": source,
            "latency_ms": ms,
        })

    st.rerun()