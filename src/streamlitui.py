"""
Motor & Cyber Law Advisor — Chat Interface
Run: streamlit run src/streamlitui.py
"""

import sys
import time
import asyncio
import logging
from pathlib import Path

import streamlit as st

# Fix event-loop issue on Windows with Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

sys.path.insert(0, str(Path(__file__).parent))

from retrievalchain import (
    load_vector_store,
    build_llm,
    init_db,
    DISCLAIMER,
    retrieve_chunks,
    format_context,
    generate_answer,
    log_query,
)

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Law Advisor",
    page_icon="⚖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0D1117; }
#MainMenu, footer, header { visibility: hidden; }

.title-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 24px 0 8px 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 8px;
}
.title-bar h1 { font-size: 20px; font-weight: 700; color: #E6EDF3; margin: 0; }
.title-badge {
    background: rgba(0,210,160,0.12);
    border: 1px solid rgba(0,210,160,0.3);
    color: #00D2A0;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 99px;
}

.msg-user { display: flex; justify-content: flex-end; margin: 4px 0; }
.msg-user .bubble {
    background: #1F6FEB; color: #FFFFFF;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%; font-size: 14px; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(31,111,235,0.3);
}
.msg-bot { display: flex; justify-content: flex-start; margin: 4px 0; }
.msg-bot .bubble {
    background: #161B22; border: 1px solid #30363D; color: #C9D1D9;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%; font-size: 14px; line-height: 1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.msg-bot.disclaimer .bubble { border-color: #F85149; color: #FFA198; }

.source-line { font-size: 11px; color: #484F58; margin: 2px 0 8px 16px; }
.source-pill {
    display: inline-block;
    background: rgba(0,210,160,0.1); border: 1px solid rgba(0,210,160,0.25);
    color: #00D2A0; font-size: 10px; font-weight: 600;
    padding: 1px 8px; border-radius: 99px; margin-right: 4px;
}

/* Regular buttons (suggestions, clear) */
.stButton > button {
    background: #161B22;
    color: #8B949E;
    font-weight: 500;
    border-radius: 10px;
    border: 1px solid #30363D;
    font-size: 13px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    border-color: #1F6FEB;
    color: #E6EDF3;
    transform: translateY(-1px);
}

hr { border-color: #21262D !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_resources():
    db            = load_vector_store()
    llm, provider = build_llm()
    conn          = init_db()
    return db, llm, conn, provider


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────
resource_ok  = False
llm_provider = "groq"
try:
    with st.spinner("Loading legal database…"):
        db, llm, conn, llm_provider = load_all_resources()
    resource_ok = True
except Exception as e:
    st.error(f"**Resource load failed:** {e}")
    st.info("Run `python src/ingestion.py` first to build the ChromaDB index.")


import base64
import os

# ─────────────────────────────────────────────────────────────────
#  TITLE BAR
# ─────────────────────────────────────────────────────────────────
provider_label = "Groq · Llama 3.3" if llm_provider == "groq" else "Gemini 2.5 Pro"

logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="50" style="border-radius: 4px; object-fit: contain;">'
else:
    logo_html = '<span style="font-size:24px">⚖</span>'

st.markdown(f"""
<div class="title-bar">
    {logo_html}
    <h1>Motor &amp; Cyber Law Advisor</h1>
    <span class="title-badge">Motor Vehicles Act · IT Act 2000</span>
    <span class="title-badge" style="margin-left:4px;background:rgba(139,92,246,0.12);border-color:rgba(139,92,246,0.3);color:#a78bfa">⚡ {provider_label}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  STATE MANAGEMENT: Extract requested query
# ─────────────────────────────────────────────────────────────────
query_to_run = None


# ─────────────────────────────────────────────────────────────────
#  CHAT HISTORY
# ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    elif msg["role"] == "assistant":
        is_disc  = not msg.get("answer_found", 1)
        disc_cls = "disclaimer" if is_disc else ""
        st.markdown(
            f'<div class="msg-bot {disc_cls}"><div class="bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
        if msg.get("chunks"):
            seen, pills = set(), ""
            for chunk in msg["chunks"]:
                code = chunk.metadata.get("act_code", "")
                name = chunk.metadata.get("act_name", code)
                if code and code not in seen:
                    seen.add(code)
                    pills += f'<span class="source-pill">{name}</span>'
            if pills:
                st.markdown(
                    f'<div class="source-line">{pills} · {msg.get("latency_ms","?")} ms</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────
#  SUGGESTED QUESTIONS  (only when no messages yet)
# ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#484F58;font-size:13px;text-align:center'>Try asking one of these:</p>",
        unsafe_allow_html=True,
    )
    suggestions = [
        "Fine for drunk driving?",
        "Penalty for no driving licence?",
        "What is Section 66C IT Act?",
        "Is hacking a crime in India?",
        "Can police seize my vehicle?",
        "What is cyber terrorism?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        with cols[i % 3]:
            # Setting query_to_run directly from button click! No need to hit Send.
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                query_to_run = q


# ─────────────────────────────────────────────────────────────────
#  CHAT INPUT (Native Streamlit chat input pinned to bottom)
# ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a legal question… (e.g. Fine for drunk driving?)", disabled=not resource_ok):
    query_to_run = prompt


# ─────────────────────────────────────────────────────────────────
#  CLEAR BUTTON
# ─────────────────────────────────────────────────────────────────
if st.session_state.messages and not query_to_run:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([4, 2, 4])
    with mid:
        # Subtle clear button
        if st.button("🗑 Clear chat", key="clear_btn"):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────
#  PROCESS QUERY
# ─────────────────────────────────────────────────────────────────
if query_to_run and resource_ok:
    # 1. Render user bubble immediately
    st.markdown(
        f'<div class="msg-user"><div class="bubble">{query_to_run}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": query_to_run})

    # 2. Show spinner while fetching
    with st.spinner("Searching legal database…"):
        _start       = time.time()
        chunks       = retrieve_chunks(db, query_to_run, k=3)
        context      = format_context(chunks)
        answer       = generate_answer(llm, context, query_to_run)
        log_query(conn, query_to_run, answer, chunks, _start)
        latency_ms   = int((time.time() - _start) * 1000)
        answer_found = 0 if DISCLAIMER in answer else 1

    # 3. Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "latency_ms": latency_ms,
        "answer_found": answer_found,
    })
    
    # 4. Rerun so chat input resets and history correctly re-renders everything
    st.rerun()