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
from auth import check_auth_callback, get_google_auth_url

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="GatiNeeti",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending" not in st.session_state:
    st.session_state.pending = None

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ── Query Params Parser (Handles Clicks/Theme/Auth) ───────────────────
if "q" in st.query_params:
    query = st.query_params["q"]
    st.session_state.pending = query
    st.query_params.clear()
    st.rerun()

if "toggle_theme" in st.query_params:
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.query_params.clear()
    st.rerun()

if "logout" in st.query_params:
    st.session_state.user = None
    st.query_params.clear()
    st.rerun()

# ── Authentication callback check ─────────────────────────────────────
check_auth_callback()

# ── Header Rendering ──────────────────────────────────────────────────
def render_custom_header():
    user_info = st.session_state.user
    theme = st.session_state.theme
    
    # Theme toggle icon & url
    theme_icon = "🌙" if theme == "light" else "☀️"
    
    # Auth button
    if user_info:
        auth_action_html = f'<a href="?logout=1" target="_self" class="google-signin-btn">Logout</a>'
    else:
        auth_url = get_google_auth_url()
        if auth_url:
            auth_action_html = f'''
            <a href="{auth_url}" target="_self" class="google-signin-btn">
                <img src="https://www.google.com/favicon.ico" style="width: 14px; height: 14px;">
                Sign In
            </a>
            '''
        else:
            auth_action_html = ''
            
    header_html = f'''
    <div class="header-container">
        <div class="header-content">
            <a href="/" target="_self" class="header-logo">GatiNeeti</a>
            <div class="header-actions">
                {auth_action_html}
                <a href="?toggle_theme=1" target="_self" class="theme-toggle-btn">{theme_icon}</a>
            </div>
        </div>
    </div>
    '''
    st.html(header_html)

render_custom_header()

# ── Custom CSS Stylesheet ─────────────────────────────────────────────
COMMON_CSS = r"""
/* Base Reset & Fonts */
body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* Hide streamlit default header and footer */
header, [data-testid="stHeader"], [data-testid="stAppHeader"], .stDeployButton {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
}
footer {
    visibility: hidden !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Page spacing and main container */
div.block-container {
    padding-top: 120px !important;
    padding-bottom: 120px !important;
    max-width: 960px !important;
}

.header-container {
    width: 100%;
    padding: 16px 40px;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 999;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.header-content {
    max-width: 1000px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-logo {
    font-family: 'Outfit', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    text-decoration: none !important;
    letter-spacing: -0.5px !important;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
}

.google-signin-btn {
    padding: 8px 16px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    text-decoration: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    transition: all 0.2s ease !important;
}

.theme-toggle-btn {
    background: none !important;
    border: none !important;
    cursor: pointer !important;
    font-size: 18px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    transition: background-color 0.2s ease !important;
    text-decoration: none !important;
}

.profile-icon {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.profile-icon img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
}

/* FAQ layout styling */
.faq-container {
    text-align: center;
    margin-top: 40px;
    margin-bottom: 40px;
}

.faq-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    margin-bottom: 40px !important;
}

.faq-subtitle {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    display: inline-block !important;
    margin-bottom: 40px !important;
}

.faq-subtitle:hover {
    text-decoration: underline !important;
}

.faq-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 20px !important;
}

@media (max-width: 900px) {
    .faq-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}

@media (max-width: 600px) {
    .faq-grid {
        grid-template-columns: 1fr !important;
    }
}

.faq-card {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    text-decoration: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    min-height: 84px !important;
    line-height: 1.4 !important;
}

.faq-card:hover {
    transform: translateY(-2px) !important;
}

.faq-arrow {
    font-size: 14px !important;
    font-weight: 700 !important;
    margin-left: 12px !important;
    transition: transform 0.2s ease !important;
}

.faq-card:hover .faq-arrow {
    transform: translateX(3px) !important;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 30px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 90% !important;
    max-width: 800px !important;
    z-index: 998 !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-testid="stChatInputContainer"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[data-testid="stChatInput"] > div {
    border-radius: 50px !important;
    padding: 8px 14px !important;
    display: flex !important;
    align-items: center !important;
}

div[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    padding: 10px 10px 10px 42px !important;
    resize: none !important;
    line-height: 1.5 !important;
    height: 44px !important;
    overflow-y: hidden !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='%232563EB' class='w-6 h-6'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.637 10.637Z'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: left 12px center !important;
    background-size: 20px 20px !important;
}

div[data-testid="stChatInput"] button {
    border-radius: 50% !important;
    width: 38px !important;
    height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    color: #FFFFFF !important;
    transition: background-color 0.2s ease, transform 0.2s ease !important;
    margin-left: 10px !important;
    padding: 0 !important;
}

div[data-testid="stChatInput"] button:hover {
    transform: scale(1.05) !important;
}

div[data-testid="stChatInput"] button svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    width: 18px !important;
    height: 18px !important;
}

/* Chat bubble styling */
div[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 16px !important;
}
"""

LIGHT_CSS = r"""
/* Light theme overrides */
.stApp {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

.header-container {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
}

.header-logo {
    color: #1E40AF !important;
}

.google-signin-btn {
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
}

.google-signin-btn:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
}

.theme-toggle-btn {
    color: #475569 !important;
}

.theme-toggle-btn:hover {
    background-color: #F1F5F9 !important;
}

.profile-icon {
    background-color: #2563EB;
    color: #FFFFFF;
}

.faq-title {
    color: #0F172A !important;
}

.faq-subtitle {
    color: #2563EB !important;
}

.faq-card {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #1E293B !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 4px 12px rgba(0,0,0,0.03) !important;
}

.faq-card:hover {
    border-color: #3B82F6 !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.08) !important;
}

.faq-arrow {
    color: #2563EB !important;
}

div[data-testid="stChatInput"] > div {
    border: 1px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #0F172A !important;
}

div[data-testid="stChatInput"] button {
    background-color: #0F52BA !important;
}

div[data-testid="stChatInput"] button:hover {
    background-color: #0a439b !important;
}

div[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
}
"""

DARK_CSS = r"""
/* Dark theme overrides */
.stApp {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
}

.header-container {
    background-color: #0F172A;
    border-bottom: 1px solid #1E293B;
}

.header-logo {
    color: #60A5FA !important;
}

.google-signin-btn {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
}

.google-signin-btn:hover {
    background-color: #334155 !important;
    border-color: #475569 !important;
}

.theme-toggle-btn {
    color: #94A3B8 !important;
}

.theme-toggle-btn:hover {
    background-color: #1E293B !important;
}

.profile-icon {
    background-color: #1E293B;
    border: 1px solid #334155;
    color: #60A5FA;
}

.faq-title {
    color: #F8FAFC !important;
}

.faq-subtitle {
    color: #60A5FA !important;
}

.faq-card {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F8FAFC !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2), 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

.faq-card:hover {
    border-color: #3B82F6 !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2) !important;
}

.faq-arrow {
    color: #60A5FA !important;
}

div[data-testid="stChatInput"] > div {
    border: 1px solid #334155 !important;
    background-color: #1E293B !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #F8FAFC !important;
}

div[data-testid="stChatInput"] button {
    background-color: #2563EB !important;
}

div[data-testid="stChatInput"] button:hover {
    background-color: #3B82F6 !important;
}

div[data-testid="stChatMessage"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}
"""

theme = st.session_state.theme
selected_css = DARK_CSS if theme == "dark" else LIGHT_CSS

st.html(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{COMMON_CSS}
{selected_css}
</style>
""")




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
        st.markdown("<h1 style='text-align:center;'>GatiNeeti</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Loading GatiNeeti...</p>", unsafe_allow_html=True)
        st.write("") # Spacer

    db, llm, conn = load_resources()
    placeholder.empty()
    resource_ok = True
except Exception as e:
    st.error(f"Failed to load database: {e}")

# ── Title ─────────────────────────────────────────────────────────────
# ── Title & Landing Page ────────────────────────────────────────────────
# Check if there is any active message or query in progress
has_messages = (
    len(st.session_state.messages) > 0
    or st.session_state.pending is not None
    or ("chat_input_val" in st.session_state and st.session_state.chat_input_val is not None)
)

if not has_messages and resource_ok:
    faq_html = """
    <div class="faq-container">
        <h1 class="faq-title">Frequently Asked Questions</h1>
        <div class="faq-grid">
            <a href="?q=What+is+the+fine+for+drunk+driving+in+India%3F" target="_self" class="faq-card">
                <span class="faq-text">What is the fine for drunk driving in India?</span>
                <span class="faq-arrow">❯</span>
            </a>
            <a href="?q=Can+I+drive+without+a+driving+licence%3F" target="_self" class="faq-card">
                <span class="faq-text">Can I drive without a driving licence?</span>
                <span class="faq-arrow">❯</span>
            </a>
            <a href="?q=Is+Section+66A+of+the+IT+Act+still+valid%3F" target="_self" class="faq-card">
                <span class="faq-text">Is Section 66A of the IT Act still valid?</span>
                <span class="faq-arrow">❯</span>
            </a>
            <a href="?q=What+is+the+punishment+for+identity+theft%3F" target="_self" class="faq-card">
                <span class="faq-text">What is the punishment for identity theft?</span>
                <span class="faq-arrow">❯</span>
            </a>
            <a href="?q=What+is+the+penalty+for+overspeeding%3F" target="_self" class="faq-card">
                <span class="faq-text">What is the penalty for overspeeding?</span>
                <span class="faq-arrow">❯</span>
            </a>
            <a href="?q=How+do+I+report+a+cyber+crime+in+India%3F" target="_self" class="faq-card">
                <span class="faq-text">How do I report a cyber crime in India?</span>
                <span class="faq-arrow">❯</span>
            </a>
        </div>
    </div>
    """
    st.html(faq_html)

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
    if st.button("Clear Chat", key="clear"):
        st.session_state.messages.clear()
        st.rerun()

# ── Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a legal question...", key="chat_input_val", disabled=not resource_ok)

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
    
    # No dynamic renaming of chat since there is no sidebar history
    
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