"""
╔══════════════════════════════════════════════════════════════════╗
║   MOTOR & CYBER LAW ADVISOR — FULL STREAMLIT APPLICATION        ║
║   Tabs: Chat · Re-Index · Evaluation · Data Explorer            ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    streamlit run src/streamlitui.py
"""

import sys
import re
import time
import csv
import asyncio
import sqlite3
import logging
import subprocess
from io import StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# Fix event-loop issue on Windows with Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Make sure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from retrievalchain import (
    load_vector_store,
    build_llm,
    init_db,
    rag_query,
    DISCLAIMER,
    GEMINI_MODEL,
    CHROMA_DIR,
    DB_LOG_PATH,
)

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor & Cyber Law Advisor",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Base ── */
.stApp { background: linear-gradient(135deg, #0A1628 0%, #0D1B2A 60%, #0A1F35 100%); }
.stSidebar { background: linear-gradient(180deg, #0F1E30 0%, #131F30 100%); border-right: 1px solid #1E3A5F; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #FFFFFF !important; font-weight: 700; }
.stMarkdown, .stText, p, li { color: #CBD5E1; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00F5C4, #00C9A7) !important;
    color: #0A1628 !important;
}

/* ── Input ── */
.stTextInput > div > div > input, .stTextArea textarea {
    background: #1A2B3C;
    color: #FFFFFF;
    border: 1px solid #2A4060;
    border-radius: 10px;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #00F5C4 !important;
    box-shadow: 0 0 0 2px rgba(0,245,196,0.15);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00F5C4, #00C9A7);
    color: #0A1628;
    font-weight: 700;
    border-radius: 10px;
    border: none;
    padding: 0.55rem 1.5rem;
    transition: all 0.2s;
    font-size: 14px;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,245,196,0.35);
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #00F5C4 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94A3B8 !important; }

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1A2B3C, #1E3347);
    border-left: 4px solid #00F5C4;
    padding: 14px 18px;
    border-radius: 0 12px 12px 0;
    margin: 10px 0;
    color: #FFFFFF;
    font-size: 15px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    animation: fadeIn 0.3s ease;
}
.bot-bubble {
    background: linear-gradient(135deg, #131F30, #162238);
    border-left: 4px solid #7B68EE;
    padding: 14px 18px;
    border-radius: 0 12px 12px 0;
    margin: 10px 0;
    color: #E2E8F0;
    font-size: 15px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    animation: fadeIn 0.3s ease;
    line-height: 1.7;
}
.disclaimer-bubble {
    background: linear-gradient(135deg, #1A0808, #200A0A);
    border-left: 4px solid #FF6B6B;
    padding: 14px 18px;
    border-radius: 0 12px 12px 0;
    margin: 10px 0;
    color: #FCA5A5;
    font-size: 15px;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

/* ── Badges ── */
.source-badge {
    display: inline-block;
    background: rgba(0,245,196,0.1);
    border: 1px solid #00F5C4;
    color: #00F5C4;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 99px;
    margin: 3px 2px;
}
.act-badge-mva {
    background: rgba(59,130,246,0.15);
    border: 1px solid #3B82F6;
    color: #93C5FD;
}
.act-badge-ita {
    background: rgba(168,85,247,0.15);
    border: 1px solid #A855F7;
    color: #D8B4FE;
}
.meta-row { font-size: 11px; color: #4A6080; margin-top: 8px; }

/* ── Cards ── */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0;
}
.success-card {
    background: rgba(0,245,196,0.06);
    border: 1px solid rgba(0,245,196,0.3);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #A7F3D0;
}
.warning-card {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #FDE68A;
}

/* ── Dividers ── */
hr { border-color: #1E3A5F !important; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #CBD5E1 !important; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CACHED RESOURCES  (load once per session)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_resources():
    db   = load_vector_store()
    llm  = build_llm()
    conn = init_db()
    return db, llm, conn


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖ Motor & Cyber Law Advisor")
    st.markdown("*Powered by Gemini + ChromaDB*")
    st.markdown("---")

    st.markdown("### 📚 Acts Covered")
    st.markdown("""
    🔵 **Motor Vehicles Act 2019**
    🟣 **IT Act 2000 + 2008 Amendment**
    """)

    st.markdown("---")
    st.markdown("### ❓ Example Questions")
    example_qs = [
        "Fine for drunk driving?",
        "Penalty for no driving licence?",
        "Identity theft punishment IT Act?",
        "What is Section 66C?",
        "Can police seize my vehicle?",
        "Is hacking a crime in India?",
        "What is cyber terrorism?",
        "Helmet rules for bikers?",
    ]
    for q in example_qs:
        if st.button(q, key=f"eq_{q[:25]}", use_container_width=True):
            st.session_state["prefill"] = q
            st.session_state["active_tab"] = 0

    st.markdown("---")
    st.markdown("### ⚙ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 3)
    st.caption(f"🤖 Model: `{GEMINI_MODEL}`")
    st.caption("⚠ Legal information only — not legal advice.")


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("# ⚖ Motor & Cyber Law Advisor")
st.markdown("Your AI-powered guide to Indian Motor Vehicles Act 2019 and IT Act 2000.")
st.markdown("---")

# Session state init
for key, default in [
    ("messages", []),
    ("prefill", ""),
    ("last_eval_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load resources
resource_ok = False
try:
    with st.spinner("⚡ Loading legal database..."):
        db, llm, conn = load_all_resources()
    resource_ok = True
except Exception as e:
    st.error(f"**Resource load failed:** {e}")
    st.info("Run `python src/ingestion.py` first to build the ChromaDB index.")


# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab_chat, tab_reindex, tab_eval, tab_data, tab_history = st.tabs([
    "💬 Chat",
    "🔄 Re-Index Data",
    "🧪 Evaluation",
    "📊 Data Explorer",
    "📋 Query History",
])


# ───────────────────────────────────────────────────────────────────
#  TAB 1 — CHAT
# ───────────────────────────────────────────────────────────────────
with tab_chat:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask your legal question:",
            value=st.session_state.get("prefill", ""),
            placeholder="e.g. What is the fine for driving without a licence?",
            key="question_input",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_btn = st.button("Ask ⚖", use_container_width=True, disabled=not resource_ok, key="ask_btn")

    # Clear prefill
    if st.session_state.get("prefill"):
        st.session_state["prefill"] = ""

    # Process query
    if (ask_btn or user_question) and user_question.strip() and resource_ok:
        question = user_question.strip()
        st.session_state["messages"].append({"role": "user", "content": question})

        with st.spinner("🔍 Searching legal database..."):
            from retrievalchain import retrieve_chunks, format_context, generate_answer, log_query
            import time as _time
            _start = _time.time()
            chunks  = retrieve_chunks(db, question, k=top_k)
            context = format_context(chunks)
            answer  = generate_answer(llm, context, question)
            log_query(conn, question, answer, chunks, _start)
            latency_ms   = int((_time.time() - _start) * 1000)
            answer_found = 0 if DISCLAIMER in answer else 1

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "latency_ms": latency_ms,
            "answer_found": answer_found,
        })

    # Render chat history
    st.markdown("---")
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            is_disc = not msg.get("answer_found", 1)
            bubble_cls = "disclaimer-bubble" if is_disc else "bot-bubble"
            icon = "⚠" if is_disc else "⚖"
            st.markdown(
                f'<div class="{bubble_cls}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Source badges
            if msg.get("chunks"):
                seen, html = set(), ""
                for chunk in msg["chunks"]:
                    code = chunk.metadata.get("act_code", "")
                    name = chunk.metadata.get("act_name", code)
                    src  = chunk.metadata.get("source", "")
                    if code not in seen:
                        seen.add(code)
                        badge_cls = "act-badge-mva" if code == "MVA" else "act-badge-ita"
                        html += f'<span class="source-badge {badge_cls}">{name}</span>'
                    if src == "scraped_toc" and "TOC" not in seen:
                        seen.add("TOC")
                        html += '<span class="source-badge">📋 Scraped TOC</span>'
                st.markdown(
                    f'<div class="meta-row">Sources: {html} &nbsp;|&nbsp; '
                    f'Latency: {msg.get("latency_ms","?")}ms</div>',
                    unsafe_allow_html=True,
                )

            # Expandable chunks
            if msg.get("chunks") and not is_disc:
                with st.expander("📄 View retrieved legal sections"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        act = chunk.metadata.get("act_name", "?")
                        src = chunk.metadata.get("source", "pdf")
                        st.markdown(f"**{i}. {act}** `[{src}]`")
                        st.code(chunk.page_content[:600], language=None)

    # Metrics + clear
    if st.session_state["messages"]:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        total    = len([m for m in st.session_state["messages"] if m["role"] == "user"])
        answered = len([m for m in st.session_state["messages"]
                        if m["role"] == "assistant" and m.get("answer_found", 0)])
        c1.metric("Queries", total)
        c2.metric("Cited answers", answered)
        c3.metric("Disclaimers", total - answered)
        c4.metric("Acts indexed", "2")

        if st.button("🗑 Clear conversation", key="clear_btn"):
            st.session_state["messages"] = []
            st.rerun()


# ───────────────────────────────────────────────────────────────────
#  TAB 2 — RE-INDEX DATA
# ───────────────────────────────────────────────────────────────────
with tab_reindex:
    st.markdown("## 🔄 Re-Index Legal Data")
    st.markdown("""
    This tab runs the full **Stage 1 Ingestion Pipeline**:
    1. Extracts text from the two PDFs
    2. Loads structured sections from the scraped CSV
    3. Cleans, chunks, and vectorises everything into ChromaDB
    """)

    # Show current DB stats
    st.markdown("### Current Database Status")
    if resource_ok:
        try:
            count = db._collection.count()
            csv_path = Path("data/raw/scraped/combined_legal_toc.csv")
            csv_rows = len(pd.read_csv(csv_path)) if csv_path.exists() else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks in ChromaDB", f"{count:,}")
            c2.metric("Scraped CSV rows", f"{csv_rows:,}")
            c3.metric("Source PDFs", "2")
        except Exception as e:
            st.warning(f"Could not read DB stats: {e}")
    else:
        st.warning("Database not loaded — run ingestion first.")

    st.markdown("---")
    st.markdown("### Source Files")

    col_a, col_b, col_c = st.columns(3)
    for col, path, label in [
        (col_a, "data/raw/it_act.pdf",                    "IT Act PDF"),
        (col_b, "data/raw/motor_vehicle.pdf",              "MVA PDF"),
        (col_c, "data/raw/scraped/combined_legal_toc.csv", "Scraped CSV"),
    ]:
        p = Path(path)
        exists = p.exists()
        size   = f"{p.stat().st_size/1024:.0f} KB" if exists else "missing"
        col.markdown(
            f'<div class="{"success-card" if exists else "warning-card"}">'
            f'{"✅" if exists else "❌"} <b>{label}</b><br>'
            f'<small>{path}<br>{size}</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── CSV ingestion toggle ──
    ingest_csv = st.checkbox(
        "Also ingest scraped CSV sections (recommended — improves section-number queries)",
        value=True,
    )

    st.markdown("### Run Ingestion")
    st.warning("⚠ Running this will **rebuild ChromaDB from scratch**. All existing vectors will be replaced.")

    if st.button("🚀 Run Full Re-Index", key="reindex_btn", disabled=not Path("data/raw/it_act.pdf").exists()):
        log_area = st.empty()
        progress = st.progress(0, text="Starting ingestion...")

        def stream_ingestion():
            # Run ingestion.py as subprocess to capture live output
            cmd = [sys.executable, str(Path(__file__).parent / "ingestion.py")]
            env_extra = {"INGEST_CSV": "1" if ingest_csv else "0"}
            import os
            proc_env = {**os.environ, **env_extra}

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).parent.parent),
                env=proc_env,
            )
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_area.code("\n".join(lines[-30:]), language=None)  # show last 30 lines
            proc.wait()
            return proc.returncode

        with st.spinner("⚙ Running ingestion pipeline..."):
            ret = stream_ingestion()

        progress.progress(100, text="Done!")
        if ret == 0:
            st.success("✅ Re-indexing complete! The database has been rebuilt.")
            st.cache_resource.clear()
            st.info("Click **Chat** tab to start querying. Resources will reload automatically.")
        else:
            st.error("❌ Ingestion failed — check the log above for errors.")


# ───────────────────────────────────────────────────────────────────
#  TAB 3 — EVALUATION
# ───────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("## 🧪 Pipeline Evaluation")
    st.markdown("""
    Run the **57-question test suite** to measure RAG pipeline quality.
    Results are categorised by topic area and exported to `data/eval_results.csv`.
    """)

    st.markdown("### Evaluation Categories")
    cats = {
        "MVA_Licence": "Driving licence rules", "MVA_Penalties": "Traffic penalties",
        "MVA_Insurance": "Vehicle insurance",   "MVA_Registration": "Vehicle registration",
        "MVA_Accident": "Road accidents",       "MVA_Misc": "MVA miscellaneous",
        "ITA_CyberCrime": "Cyber crimes",       "ITA_Esign": "E-signatures",
        "ITA_Intermediary": "Intermediaries",   "ITA_66A": "Section 66A status",
        "OOS": "Out-of-scope (disclaimer)",
    }
    cols = st.columns(4)
    for i, (cat, desc) in enumerate(cats.items()):
        cols[i % 4].markdown(f"- **{cat}**: {desc}")

    st.markdown("---")
    st.info("⏱ Full evaluation takes ~15 minutes (57 queries × 13s rate-limit delay on free tier).")

    # Load previous results if available
    eval_csv_path = Path("data/eval_results.csv")
    if eval_csv_path.exists():
        prev_df = pd.read_csv(eval_csv_path)
        st.markdown("### Last Evaluation Results")
        total   = len(prev_df)
        passed  = (prev_df["status"] == "PASS").sum()
        failed  = (prev_df["status"] == "FAIL").sum()
        errors  = (prev_df["status"] == "ERROR").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("Passed ✓", f"{passed} ({100*passed//total}%)")
        c3.metric("Failed ✗", f"{failed} ({100*failed//total}%)")
        c4.metric("Errors", errors)

        # Category breakdown
        cat_df = (prev_df.groupby("category")["status"]
                  .value_counts().unstack(fill_value=0).reset_index())
        if "PASS" in cat_df and "FAIL" in cat_df:
            cat_df["Rate"] = (cat_df["PASS"] / (cat_df["PASS"] + cat_df.get("FAIL", 0)) * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(cat_df, use_container_width=True)

        # Show failed queries
        failed_df = prev_df[prev_df["status"] != "PASS"]
        if not failed_df.empty:
            with st.expander(f"⚠ View {len(failed_df)} failed/error queries"):
                st.dataframe(failed_df[["category","question","status","note"]], use_container_width=True)

        # Download
        st.download_button(
            "⬇ Download eval_results.csv",
            data=eval_csv_path.read_bytes(),
            file_name="eval_results.csv",
            mime="text/csv",
        )

    if st.button("▶ Run Full Evaluation Now", key="run_eval_btn", disabled=not resource_ok):
        st.warning("Running evaluation in background — do not close this tab.")
        log_out = st.empty()
        prog    = st.progress(0)

        from evaluate import TEST_SUITE, RATE_LIMIT_DELAY_SEC

        results = []
        for i, (question, expected_act, category) in enumerate(TEST_SUITE, 1):
            prog.progress(i / len(TEST_SUITE), text=f"[{i}/{len(TEST_SUITE)}] {question[:60]}...")

            if i > 1 and RATE_LIMIT_DELAY_SEC > 0:
                time.sleep(RATE_LIMIT_DELAY_SEC)

            try:
                result       = rag_query(db, llm, conn, question)
                answer_found = result["answer_found"]
                latency_ms   = result["latency_ms"]
                top_act      = result["chunks"][0].metadata.get("act_code", "?") if result["chunks"] else "—"

                if expected_act is None:
                    status = "PASS" if answer_found == 0 else "FAIL"
                    note   = "Disclaimer returned" if status == "PASS" else "Expected disclaimer"
                else:
                    status = "PASS" if answer_found == 1 else "FAIL"
                    note   = "Cited answer" if status == "PASS" else "Expected cited answer"
            except Exception as exc:
                status, note, latency_ms, top_act = "ERROR", str(exc), 0, "—"

            results.append({
                "index": i, "category": category, "question": question,
                "expected_act": expected_act or "OOS", "status": status,
                "latency_ms": latency_ms, "top_act": top_act, "note": note,
            })
            log_out.dataframe(pd.DataFrame(results).tail(10), use_container_width=True)

        prog.progress(100, text="Evaluation complete!")
        res_df = pd.DataFrame(results)
        res_df.to_csv("data/eval_results.csv", index=False)
        st.success(f"✅ Done! {(res_df['status']=='PASS').sum()}/{len(res_df)} passed.")
        st.rerun()


# ───────────────────────────────────────────────────────────────────
#  TAB 4 — DATA EXPLORER
# ───────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("## 📊 Data Explorer")

    sub = st.radio("View:", ["Scraped CSV Sections", "ChromaDB Chunks", "Source PDFs"], horizontal=True)

    if sub == "Scraped CSV Sections":
        csv_path = Path("data/raw/scraped/combined_legal_toc.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            st.markdown(f"**{len(df)} section entries** scraped from IndiaCode.gov.in")

            act_filter = st.multiselect("Filter by Act", df["act_code"].unique().tolist(), default=df["act_code"].unique().tolist())
            show_repealed = st.checkbox("Show repealed sections", value=False)

            filtered = df[df["act_code"].isin(act_filter)]
            if not show_repealed:
                filtered = filtered[filtered["is_repealed"] == False]

            search = st.text_input("🔍 Search section titles", placeholder="e.g. helmet, identity theft")
            if search:
                filtered = filtered[filtered["section_title"].str.contains(search, case=False, na=False)]

            st.dataframe(
                filtered[["act_code", "section_number", "section_title", "is_repealed", "url"]],
                use_container_width=True,
                height=500,
            )
            st.download_button(
                "⬇ Download filtered CSV",
                data=filtered.to_csv(index=False),
                file_name="filtered_sections.csv",
                mime="text/csv",
            )
        else:
            st.error("CSV not found. Run `python src/webscraper.py` first.")

    elif sub == "ChromaDB Chunks":
        if resource_ok:
            st.markdown("### Semantic Search on ChromaDB")
            search_q = st.text_input("Search ChromaDB:", placeholder="e.g. Section 66C identity theft")
            k_search = st.slider("Number of results", 1, 20, 5)

            if search_q:
                results = db.similarity_search_with_score(search_q, k=k_search)
                for i, (doc, score) in enumerate(results, 1):
                    act   = doc.metadata.get("act_name", "?")
                    code  = doc.metadata.get("act_code", "")
                    src   = doc.metadata.get("source", "pdf")
                    badge = "🔵" if code == "MVA" else "🟣"
                    with st.expander(f"{badge} [{i}] {act} — score {score:.3f} `[{src}]`"):
                        st.code(doc.page_content, language=None)
                        st.json(doc.metadata)
        else:
            st.error("Database not loaded.")

    elif sub == "Source PDFs":
        st.markdown("### Source PDF Files")
        for pdf_path in ["data/raw/it_act.pdf", "data/raw/motor_vehicle.pdf"]:
            p = Path(pdf_path)
            if p.exists():
                size_mb = p.stat().st_size / 1024 / 1024
                st.markdown(
                    f'<div class="success-card">✅ <b>{p.name}</b> — {size_mb:.1f} MB</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="warning-card">❌ <b>{Path(pdf_path).name}</b> — not found</div>',
                    unsafe_allow_html=True,
                )

        # Processed text files
        st.markdown("### Cleaned Text Files")
        for txt_path in ["data/processed/ita_clean.txt", "data/processed/mva_clean.txt"]:
            p = Path(txt_path)
            if p.exists():
                size_kb = p.stat().st_size / 1024
                with st.expander(f"📄 {p.name} ({size_kb:.0f} KB)"):
                    st.text(p.read_text(encoding="utf-8")[:3000] + "\n\n... [truncated]")


# ───────────────────────────────────────────────────────────────────
#  TAB 5 — QUERY HISTORY
# ───────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("## 📋 Query History")
    st.markdown("All queries logged to SQLite (`data/query_log.db`)")

    try:
        hist_conn = sqlite3.connect(DB_LOG_PATH)
        hist_df = pd.read_sql("SELECT * FROM query_log ORDER BY id DESC LIMIT 200", hist_conn)
        hist_conn.close()

        if hist_df.empty:
            st.info("No queries logged yet. Ask a question in the Chat tab!")
        else:
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total logged", len(hist_df))
            c2.metric("Answered", int(hist_df["answer_found"].sum()))
            c3.metric("Disclaimers", int((hist_df["answer_found"] == 0).sum()))
            avg_lat = int(hist_df["latency_ms"].mean()) if "latency_ms" in hist_df.columns else "?"
            c4.metric("Avg latency", f"{avg_lat}ms")

            st.markdown("---")

            # Search
            search_hist = st.text_input("🔍 Search history", placeholder="Filter by keyword...")
            show_df = hist_df.copy()
            if search_hist:
                show_df = show_df[show_df["user_query"].str.contains(search_hist, case=False, na=False)]

            st.dataframe(
                show_df[["id","timestamp","user_query","act_code","source_section_cited","latency_ms","answer_found"]],
                use_container_width=True,
                height=450,
            )

            # Download
            st.download_button(
                "⬇ Download query_log.csv",
                data=hist_df.to_csv(index=False),
                file_name="query_log.csv",
                mime="text/csv",
            )

            # Latency chart
            if "latency_ms" in hist_df.columns and len(hist_df) > 1:
                st.markdown("### Response Latency Trend")
                chart_df = hist_df[["id","latency_ms"]].sort_values("id")
                st.line_chart(chart_df.set_index("id"))

    except Exception as e:
        st.error(f"Could not load query history: {e}")