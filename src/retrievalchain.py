"""
╔══════════════════════════════════════════════════════════════════╗
║      MOTOR & CYBER LAW ADVISOR — STAGE 2: RETRIEVAL CHAIN       ║
║      ChromaDB → LangChain → Groq (Llama 3) → Cited Answer       ║
╚══════════════════════════════════════════════════════════════════╝

What this file does:
  1. Loads the ChromaDB vector store built in Stage 1
  2. Accepts a user query
  3. Embeds the query with the SAME HuggingFace model used in Stage 1
  4. Retrieves the top-k most relevant legal chunks (semantic search)
  5. Injects chunks into a strict citation prompt
  6. Sends prompt to Groq (Llama 3)
  7. Returns cited answer or forced disclaimer
  8. Logs every query → SQLite (for Power BI analysis)

Install:
    pip install langchain langchain-community langchain-groq
    pip install sentence-transformers chromadb
    pip install python-dotenv

.env file (create in project root):
    GROQ_API_KEY=your_key_here

Run:
    python src/retrievalchain.py
"""

# ─────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ─────────────────────────────────────────────────────────────────
#  CONFIG  ← edit these if your paths differ
# ─────────────────────────────────────────────────────────────────
CHROMA_DIR   = "data/processed/chroma_db"    # built by Stage 1
DB_LOG_PATH  = "data/query_log.db"           # SQLite log for Power BI
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"
TOP_K        = 3      # number of chunks retrieved per query
EMBED_DEVICE = "cpu"  # "cuda" if you have an Nvidia GPU

DISCLAIMER = (
    "I do not have the specific legal data to answer this. "
    "Please consult a qualified lawyer."
)

# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/stage2_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 1 — SYSTEM PROMPT
#
#  This is the exact prompt injected into Groq before every query.
#  It defines the AI's persona, citation rules, fallback behaviour,
#  and the Section 66A constitutional void notice.
#
#  {context}    → replaced with retrieved legal chunks
#  {question}   → replaced with the user's question
# ═══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
You are a Legal Information Assistant specialised in Indian law.
You help Indian citizens understand their rights under the
Motor Vehicles Act 2019 (MVA) and the Information Technology Act 2000 (ITA).

RULES:

1. ANSWER HELPFULLY
   Give a clear, accurate answer using your knowledge of MVA and ITA.
   Use the CONTEXT sections below as a reference guide — they tell you
   which sections are relevant. Fill in the details from your legal
   knowledge of these well-known Indian public laws.

2. ALWAYS CITE SECTIONS
   For every legal point, include a citation in this format:
   (Section NUMBER, Act Name)
   Example: "Drunk driving is punishable with ₹10,000 fine for first
   offence. (Section 185, Motor Vehicles Act 2019)"

3. STAY IN SCOPE
   Only answer questions about MVA 2019 and ITA 2000.
   For anything else, say: "This is outside the scope of the Motor
   Vehicles Act and IT Act. Please consult a qualified lawyer."

4. SECTION 66A IS VOID
   If asked about Section 66A of the IT Act, explain it was struck
   down by the Supreme Court in Shreya Singhal v. Union of India (2015)
   and is no longer valid law.

5. ALWAYS ADD A DISCLAIMER
   End every answer with: "Note: This is legal information, not legal
   advice. Please consult a qualified lawyer for your specific situation."

════════════════════════════════════════════════════
CONTEXT — Relevant sections identified by the database:
{context}
════════════════════════════════════════════════════

USER QUESTION: {question}
"""


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 2 — VECTOR STORE LOADER
#  Loads the ChromaDB index built in Stage 1.
#  Uses the SAME embedding model — vectors must be in same space.
# ═══════════════════════════════════════════════════════════════════
def load_vector_store() -> Chroma:
    """
    Load ChromaDB from disk.

    Returns:
        Chroma vectorstore object ready for similarity search.
    """
    log.info(f"Loading ChromaDB from: {CHROMA_DIR}")

    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Run stage1_ingestion.py first to build the vector store."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    count = db._collection.count()
    log.info(f"ChromaDB loaded — {count:,} chunks indexed")

    if count == 0:
        raise ValueError(
            "ChromaDB is empty. Stage 1 may have failed. "
            "Re-run stage1_ingestion.py."
        )

    return db


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 3 — RETRIEVER
#  Converts the user query to a vector and finds top-k similar chunks.
#
#  Why semantic search beats keyword search for law:
#  "drunk driving fine" → retrieves Section 185 MVA even though
#  the section text says "driving under influence of alcohol" —
#  no word overlap needed.
# ═══════════════════════════════════════════════════════════════════
def retrieve_chunks(db: Chroma, query: str, k: int = TOP_K) -> list:
    """
    Retrieve top-k most semantically similar chunks.

    Args:
        db:    Loaded ChromaDB vectorstore
        query: User's question in plain language
        k:     Number of chunks to retrieve

    Returns:
        List of LangChain Document objects with text + metadata.
    """
    log.info(f"Retrieving top-{k} chunks for: '{query[:80]}'")

    results = db.similarity_search_with_score(query, k=k)

    for i, (doc, score) in enumerate(results):
        act = doc.metadata.get("act_code", "?")
        log.info(f"  [{i+1}] {act} — score {score:.3f} — {doc.page_content[:60]}...")

    # Return just the documents (without scores)
    return [doc for doc, _ in results]


def format_context(chunks: list) -> str:
    """
    Format retrieved chunks into the context block injected into the prompt.
    Each chunk is labelled with its source act.
    """
    if not chunks:
        return "No relevant legal sections found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        act_name = chunk.metadata.get("act_name", "Unknown Act")
        act_code = chunk.metadata.get("act_code", "")
        parts.append(
            f"--- Source {i}: {act_name} ({act_code}) ---\n"
            f"{chunk.page_content.strip()}"
        )

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 4 — LLM CHAIN (Groq Llama 3)
#  Sends the system prompt + retrieved context + user query to Groq.
#  Returns the generated cited answer.
# ═══════════════════════════════════════════════════════════════════
def build_llm() -> ChatGroq:
    """
    Initialise Groq (Llama 3).

    Requires GROQ_API_KEY in .env file.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here"
        )

    llm = ChatGroq(
        model_name=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.0,      # 0 = deterministic, no creative guessing
        max_tokens=2048,
    )

    log.info(f"Groq model ready: {GROQ_MODEL}")
    return llm


def generate_answer(llm, context: str, question: str) -> str:
    """
    Send the filled prompt to Groq and return the response.

    Args:
        llm:      Initialised Groq LLM
        context:  Formatted retrieved chunks
        question: User's original question

    Returns:
        The LLM's response as a string.
    """
    filled_prompt = SYSTEM_PROMPT.format(
        context=context,
        question=question,
    )

    messages = [HumanMessage(content=filled_prompt)]

    log.info("Sending prompt to Groq...")
    response = llm.invoke(messages)

    return response.content.strip()


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 4b — SELF-LEARNING / AUTO-TRAIN
#
#  When a query returns a disclaimer (answer_found=0), this function:
#  1. Re-asks the LLM with an unrestricted prompt to generate an answer
#  2. Saves the Q&A as a new document in ChromaDB
#  3. Next time a similar question is asked, the DB will have the answer
# ═══════════════════════════════════════════════════════════════════
LEARN_PROMPT = """
You are an expert on Indian law, specifically the Motor Vehicles Act 2019
and the Information Technology Act 2000.

A user asked the following question and the system could not find a clear
answer in its database. Please provide a thorough, accurate answer based
on your knowledge of these laws.

Format requirements:
- Cite the exact section number(s) for every legal claim.
- Format: (Section NUMBER, Act Name)
- Keep the answer factual, clear, and under 300 words.
- End with: "Note: This is legal information, not legal advice."

QUESTION: {question}
"""


def learn_from_unanswered(db: Chroma, llm, question: str) -> str:
    """
    Auto-train: generate a proper answer for a missed question
    and add it to ChromaDB so future queries find it.

    Returns the generated answer string.
    """
    from langchain.schema import Document

    log.info(f"[AUTO-LEARN] Generating answer for missed question: '{question[:80]}'")

    # Step 1: Generate answer without context restriction
    filled = LEARN_PROMPT.format(question=question)
    messages = [HumanMessage(content=filled)]
    response = llm.invoke(messages)
    answer = response.content.strip()

    # Detect which act this is about
    q_lower = question.lower()
    if any(w in q_lower for w in ["cyber", "hacking", "it act", "identity theft",
                                   "digital", "electronic", "computer", "section 66",
                                   "section 43", "section 67", "intermediary"]):
        act_code, act_name = "ITA", "Information Technology Act 2000"
    else:
        act_code, act_name = "MVA", "Motor Vehicles Act 2019"

    # Step 2: Create document content = question + answer (boosts future retrieval)
    doc_content = f"Q: {question}\nA: {answer}"
    doc = Document(
        page_content=doc_content,
        metadata={
            "act_code":   act_code,
            "act_name":   act_name,
            "source":     "auto_learned",
            "question":   question,
        }
    )

    # Step 3: Add to ChromaDB
    db.add_documents([doc])
    log.info(f"[AUTO-LEARN] Saved to ChromaDB — act={act_code} | chars={len(doc_content)}")

    # Step 4: Persist the new document to a JSON log
    import json
    learn_log_path = Path("data/auto_learned.json")
    learn_log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if learn_log_path.exists():
        try:
            entries = json.loads(learn_log_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries.append({"question": question, "answer": answer, "act": act_code})
    learn_log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"[AUTO-LEARN] Logged to {learn_log_path}")

    return answer


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 5 — SQLITE LOGGER
#  Every query is logged for Power BI analysis.
#
#  Columns:
#    timestamp           — when the query was made
#    user_query          — raw question text
#    llm_response        — full answer from Groq
#    source_section_cited — section number extracted from answer
#    act_code            — MVA or ITA (from top retrieved chunk)
#    latency_ms          — response time in milliseconds
#    answer_found        — 1 if cited answer, 0 if disclaimer given
#
#  Power BI uses this to build:
#    • Confusion heatmap (which sections are asked about most)
#    • Coverage gap report (queries that returned no source)
#    • Latency tracker (performance over time)
# ═══════════════════════════════════════════════════════════════════
def init_db(db_path: str = DB_LOG_PATH) -> sqlite3.Connection:
    """
    Create the query_log table if it doesn't exist.
    Returns a SQLite connection.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp            TEXT    NOT NULL,
            user_query           TEXT    NOT NULL,
            llm_response         TEXT,
            source_section_cited TEXT,
            act_code             TEXT,
            latency_ms           INTEGER,
            answer_found         INTEGER
        )
    """)
    conn.commit()
    log.info(f"SQLite log ready: {db_path}")
    return conn


def log_query(conn: sqlite3.Connection, query: str, response: str,
              chunks: list, start_time: float) -> None:
    """
    Write one row to the query_log table.
    """
    latency_ms   = int((time.time() - start_time) * 1000)
    answer_found = 0 if DISCLAIMER in response else 1

    # Extract act_code from the top retrieved chunk
    act_code = chunks[0].metadata.get("act_code", "") if chunks else ""

    # Try to extract a section number from the response
    import re
    section_match = re.search(r"Section\s+(\d+[A-Za-z]?)", response)
    source_cited  = section_match.group(0) if section_match else (None if not answer_found else "cited")

    conn.execute(
        "INSERT INTO query_log VALUES (NULL,?,?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            query,
            response,
            source_cited,
            act_code,
            latency_ms,
            answer_found,
        ),
    )
    conn.commit()
    log.info(f"Logged — latency {latency_ms}ms | answer_found={answer_found} | act={act_code}")


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 6 — FULL RAG PIPELINE
#  Ties all blocks together: query → retrieve → generate → log → return
# ═══════════════════════════════════════════════════════════════════
def rag_query(db: Chroma, llm, conn: sqlite3.Connection,
              question: str) -> dict:
    """
    Complete RAG pipeline for one user query.

    Args:
        db:       Loaded ChromaDB vectorstore
        llm:      Initialised Groq LLM
        conn:     SQLite connection for logging
        question: User's plain-language legal question

    Returns:
        dict with keys: answer, chunks, latency_ms, answer_found
    """
    start = time.time()

    # Step 1 — Retrieve
    chunks  = retrieve_chunks(db, question, k=TOP_K)
    context = format_context(chunks)

    # Step 2 — Generate
    answer  = generate_answer(llm, context, question)

    # Step 3 — Log
    log_query(conn, question, answer, chunks, start)

    latency_ms   = int((time.time() - start) * 1000)
    answer_found = 0 if DISCLAIMER in answer else 1

    return {
        "question":     question,
        "answer":       answer,
        "chunks":       chunks,
        "latency_ms":   latency_ms,
        "answer_found": answer_found,
    }


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 7 — VERIFICATION (run before connecting Streamlit UI)
#  Tests the full pipeline with 4 queries — 2 MVA, 1 ITA, 1 unknown
# ═══════════════════════════════════════════════════════════════════
def verify_pipeline(db: Chroma, llm, conn: sqlite3.Connection) -> None:
    """
    Run 4 test queries to confirm Stage 2 works end-to-end.
    This is your go/no-go check before building the Streamlit UI.
    """
    test_queries = [
        # Should cite Section 185 MVA
        ("What is the fine for drunk driving in India?", "MVA"),
        # Should cite Section 4 MVA
        ("Can I drive without a driving licence?", "MVA"),
        # Should cite Section 66C ITA
        ("What is the punishment for identity theft?", "ITA"),
        # Should return disclaimer (not in documents)
        ("What are the rules for drone registration?", None),
    ]

    print("\n" + "═" * 65)
    print("  STAGE 2 VERIFICATION — 4 TEST QUERIES")
    print("═" * 65)

    all_pass = True
    for question, expected_act in test_queries:
        print(f"\nQ: {question}")
        result = rag_query(db, llm, conn, question)

        print(f"A: {result['answer'][:300]}...")
        print(f"   Latency: {result['latency_ms']}ms | "
              f"Answer found: {result['answer_found']} | "
              f"Top chunk act: {result['chunks'][0].metadata.get('act_code','?') if result['chunks'] else '—'}")

        # Check: disclaimer queries should NOT have answer_found=1
        if expected_act is None and result["answer_found"] == 1:
            print("   ⚠ Expected disclaimer but got an answer — check prompt")
            all_pass = False
        elif expected_act is not None and result["answer_found"] == 0:
            print(f"   ⚠ Expected answer but got disclaimer — check ChromaDB coverage")
            all_pass = False
        else:
            print("   ✓ PASS")

    print("\n" + "═" * 65)
    if all_pass:
        print("  All verification queries passed. Ready for Stage 3 UI.")
    else:
        print("  Some checks failed — review logs above before proceeding.")
    print("═" * 65 + "\n")


# ═══════════════════════════════════════════════════════════════════
#  INTERACTIVE QUERY LOOP (for testing without Streamlit)
# ═══════════════════════════════════════════════════════════════════
def interactive_loop(db: Chroma, llm, conn: sqlite3.Connection) -> None:
    """
    Simple terminal chat loop for testing Stage 2 before the UI.
    Type 'quit' to exit.
    """
    print("\n" + "─" * 65)
    print("  Motor & Cyber Law Advisor — Stage 2 Interactive Mode")
    print("  Type your legal question. Type 'quit' to exit.")
    print("─" * 65)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        result = rag_query(db, llm, conn, question)

        print("\n" + "─" * 55)
        print(f"Answer:\n{result['answer']}")
        print(f"\n[Latency: {result['latency_ms']}ms | "
              f"Sources retrieved: {len(result['chunks'])}]")
        print("─" * 55)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("  Motor & Cyber Law Advisor — Stage 2: Retrieval Chain")
    log.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 65)

    # Load all components
    db   = load_vector_store()
    llm  = build_llm()
    conn = init_db()

    # Run verification first
    verify_pipeline(db, llm, conn)

    # Then enter interactive mode
    interactive_loop(db, llm, conn)

    conn.close()
    log.info("Stage 2 session ended.")


if __name__ == "__main__":
    main()
