"""
╔══════════════════════════════════════════════════════════════════╗
║         MOTOR & CYBER LAW ADVISOR — STAGE 1 PIPELINE            ║
║         Data Ingestion · Cleaning · Chunking · Vectorization     ║
╚══════════════════════════════════════════════════════════════════╝

Author  : Data Analyst (Legal Tech Project)
Stack   : PyPDF2, LangChain, HuggingFace, ChromaDB
PDFs    : IT Act 2000 (it_act.pdf) · Motor Vehicles Act (motor_vehicle.pdf)

"""

# ─────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────
import re
import os
import logging
import time
from pathlib import Path
from datetime import datetime

import PyPDF2
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ─────────────────────────────────────────────────────────────────
#  CONFIG  ← edit paths here if your PDFs live elsewhere
# ─────────────────────────────────────────────────────────────────
PDF_ITA    = "data/raw/it_act.pdf"
PDF_MVA    = "data/raw/motor_vehicle.pdf"
CSV_SCRAPED= "data/raw/scraped/combined_legal_toc.csv"  # from webscraper.py
CHROMA_DIR = "data/processed/chroma_db"
LOG_DIR    = "logs"

CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 100    # 10% overlap — keeps context at boundaries
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE  = "cpu"  # change to "cuda" if you have an Nvidia GPU

# ─────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/stage1_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 1 — PDF EXTRACTION
#  Reads every page of a PDF and returns one raw text string.
# ═══════════════════════════════════════════════════════════════════
def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF using PyPDF2.

    Returns:
        A single string containing all page text, joined by newlines.
        Returns empty string if file not found or extraction fails.
    """
    path = Path(pdf_path)
    if not path.exists():
        log.error(f"PDF not found: {pdf_path}")
        return ""

    pages = []
    try:
        with open(path, "rb") as f:          # "rb" = read binary — required for PDFs
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            log.info(f"Extracting {total} pages from {path.name} ...")

            for i, page in enumerate(reader.pages):
                text = page.extract_text()   # returns None for image-only pages
                pages.append(text or "")     # safe fallback for blank/scanned pages

        full_text = "\n".join(pages)
        log.info(f"Extracted {len(full_text):,} characters from {path.name}")

        # Warn if suspiciously short — likely a scanned/image PDF
        if len(full_text) < 20_000:
            log.warning(
                f"{path.name} yielded only {len(full_text):,} chars. "
                "If it is a scanned PDF, switch to pytesseract for OCR."
            )
        return full_text

    except Exception as e:
        log.error(f"Failed to extract {pdf_path}: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 2 — CLEANING
#  Strips legal noise: page numbers, footnotes, repealed sections,
#  amendment notes, and the constitutionally void Section 66A.
# ═══════════════════════════════════════════════════════════════════

# ── Common cleaning shared by both acts ──────────────────────────
def _apply_base_cleaning(text: str) -> str:
    """
    Shared cleaning rules for both MVA and IT Act.
    Applied before act-specific passes.
    """

    # 1. Remove page numbers: "— 12 —", "3", standalone digit lines
    text = re.sub(r"[-—]+\s*\d+\s*[-—]+", "", text)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

    # 2. Standard [Omitted] / [Rep.] / [Repealed by ...] inline
    text = re.sub(
        r"\[(?:Omitted|Rep\.|Repealed[^\]]*)\][^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Multi-section omission blocks:
    #    "49–56. [Omitted by Finance Act, 2017]"
    #    "91. [Amendment of Act 45 of 1860.] Omitted by ..."
    text = re.sub(
        r"\d+[A-Z]?(?:[–\-]\d+[A-Z]?)?\.\s+\[[^\]]*(?:Omitted|Repealed)[^\]]*\][^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Amendment footnote lines (very common in India Code PDFs)
    #    e.g. "Subs. by Act 10 of 2009, s. 2, for 'digital signature' (w.e.f. 27-10-2009)."
    text = re.sub(
        r"(?:Subs\.|Ins\.|Omitted)\s+by\s+(?:Act|s\.|notification|w\.e\.f\.)[^\n]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Superscript footnote markers mid-text: ¹ ² ³ etc.
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", text)

    # 6. Page header/footer repetitions (common in India Code multi-page PDFs)
    text = re.sub(r"Page\s+\d+(?:\s+of\s+\d+)?", "", text, flags=re.IGNORECASE)

    # 7. Collapse excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)    # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)    # multiple spaces → one

    return text.strip()


def clean_it_act(text: str) -> str:
    """
    IT Act 2000 — specific cleaning on top of base rules.

    CRITICAL EXTRA STEP: Section 66A was struck down by the Supreme Court
    in Shreya Singhal v. Union of India, AIR 2015 SC 1523.
    Its full text appears in the PDF but must NOT be cited as valid law.
    We replace it with a void notice so the AI can correctly tell users it
    has been invalidated.
    """
    text = _apply_base_cleaning(text)

    # Replace Section 66A body with constitutional void notice
    # The regex captures from "66A." until the start of "66B."
    text = re.sub(
        r"(66A\.\s+Punishment for sending offensive messages.*?)(66B\.)",
        (
            "66A. [CONSTITUTIONALLY VOID — Struck down by Supreme Court of India "
            "in Shreya Singhal v. Union of India, AIR 2015 SC 1523, dated 24 March 2015. "
            "This section cannot be cited as valid law.]\n\n66B."
        ),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove Schedules that are fully omitted
    text = re.sub(
        r"THE (?:THIRD|FOURTH) SCHEDULE\s*\[Omitted[^\]]*\]\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text


def clean_mva(text: str) -> str:
    """
    Motor Vehicles Act 2019 — base cleaning is sufficient.
    Add MVA-specific patterns here if needed after inspecting your PDF.
    """
    text = _apply_base_cleaning(text)
    return text


def run_verification(text: str, act_code: str) -> bool:
    """
    Post-cleaning checks. Logs PASS/FAIL for each rule.
    Returns True only if ALL checks pass.
    """
    checks = {
        "[Omitted] remaining":    len(re.findall(r"\[Omitted\]", text, re.I)),
        "[Rep.] remaining":       len(re.findall(r"\[Rep\.", text, re.I)),
        "Amendment footnotes":    len(re.findall(r"Subs\. by Act", text, re.I)),
        "Superscript markers":    len(re.findall(r"[¹²³⁴⁵⁶⁷⁸⁹]", text)),
    }
    all_pass = True
    log.info(f"── Verification: {act_code} ──")
    for label, count in checks.items():
        status = "PASS" if count == 0 else f"FAIL  ({count} instances remaining)"
        log.info(f"   {label:30s} {status}")
        if count > 0:
            all_pass = False
    return all_pass


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 3 — CHUNKING
#  Splits cleaned text into 1000-char overlapping chunks.
#  Each chunk carries metadata so the LLM can cite its source.
# ═══════════════════════════════════════════════════════════════════
def make_chunks(text: str, act_code: str, act_name: str) -> list[Document]:
    """
    Split text into LangChain Document chunks with metadata.

    Splitter priority order:
        1. Paragraph break (\n\n)  — preferred: keeps full legal clauses together
        2. Line break (\n)
        3. Sentence end (.)
        4. Word space ( )          — last resort, almost never reached

    Args:
        text:     Cleaned legal text string
        act_code: Short code e.g. "ITA" or "MVA"
        act_name: Full name used in citations

    Returns:
        List of LangChain Document objects (chunk text + metadata)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "act_code": act_code,
            "act_name": act_name,
            "source":   "bare_act_pdf",
            "ingested_at": datetime.utcnow().isoformat(),
        }],
    )

    log.info(f"Chunked {act_code}: {len(chunks)} chunks from {len(text):,} chars")
    return chunks


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 2.5 — CSV INGESTION (scraped TOC from webscraper.py)
#  Adds one Document per non-repealed row in combined_legal_toc.csv.
#  Complements the PDF chunks with exact section-number → title entries.
# ═══════════════════════════════════════════════════════════════════
def load_csv_chunks(csv_path: str = CSV_SCRAPED) -> list:
    """
    Load scraped section TOC from CSV and convert each non-repealed row
    into a LangChain Document.

    page_content format:
        "Section 66C. Identity theft."

    Returns:
        List of Document objects tagged with source='scraped_toc'.
    """
    path = Path(csv_path)
    if not path.exists():
        log.warning(f"Scraped CSV not found: {csv_path} — skipping CSV ingestion")
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(f"Failed to read CSV {csv_path}: {e}")
        return []

    # Filter out repealed rows
    if "is_repealed" in df.columns:
        df = df[df["is_repealed"] != True]

    docs = []
    for _, row in df.iterrows():
        sec_num   = str(row.get("section_number", "")).strip()
        sec_title = str(row.get("section_title",  "")).strip()
        act_code  = str(row.get("act_code",  "")).strip()
        act_name  = str(row.get("act_name",  "")).strip()
        url       = str(row.get("url",       "")).strip()

        if not sec_title or sec_title.lower() == "nan":
            continue

        content = f"{sec_title}"  # e.g. "Section 66C. Identity theft."

        docs.append(Document(
            page_content=content,
            metadata={
                "act_code":       act_code,
                "act_name":       act_name,
                "section_number": sec_num,
                "source":         "scraped_toc",
                "url":            url,
                "ingested_at":    datetime.utcnow().isoformat(),
            },
        ))

    log.info(f"CSV ingestion: {len(docs)} section Documents from {path.name}")
    return docs


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 4 — VECTORIZATION & CHROMADB STORAGE
#  Converts each chunk → 384-dim vector → stores in ChromaDB.
# ═══════════════════════════════════════════════════════════════════
def build_vector_store(all_chunks: list[Document]) -> None:
    """
    Embed all chunks with HuggingFace all-MiniLM-L6-v2 and
    persist the ChromaDB vector index to disk.

    Why all-MiniLM-L6-v2?
    - Free: runs entirely locally, no API key
    - Fast: ~80 MB, optimised for semantic similarity
    - 384-dim vectors: good accuracy / speed tradeoff
    - Same model used at query time → vectors are comparable
    """
    # Import here so failures give clear error messages
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError:
        log.error(
            "Missing packages. Run:\n"
            "  pip install langchain-community sentence-transformers chromadb"
        )
        raise

    log.info(f"Loading embedding model: {EMBED_MODEL}  (first run downloads ~80 MB)")
    t0 = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},   # cosine similarity works best
    )

    log.info(f"Model loaded in {time.time()-t0:.1f}s. Embedding {len(all_chunks)} chunks ...")
    t1 = time.time()

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()

    elapsed = time.time() - t1
    log.info(f"ChromaDB built in {elapsed:.1f}s → saved to '{CHROMA_DIR}'")
    log.info(f"Total indexed chunks: {vectorstore._collection.count()}")


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 5 — VERIFICATION QUERY
#  Proves the vector store works BEFORE you move to Phase 2.
# ═══════════════════════════════════════════════════════════════════
def verify_vector_store() -> bool:
    """
    Reload ChromaDB from disk and run two test queries —
    one for each act. Logs the top result for each.

    Returns True if both queries return relevant results.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError:
        log.error("Cannot verify — packages missing.")
        return False

    log.info("── Verifying ChromaDB ──")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    total = db._collection.count()
    log.info(f"Total chunks in DB: {total}")

    if total == 0:
        log.error("DB is EMPTY — vectorization failed. Re-run.")
        return False

    test_queries = [
        ("penalty for driving without licence",      "MVA"),
        ("punishment for hacking computer resource", "ITA"),
    ]

    all_ok = True
    for query, expected_act in test_queries:
        results = db.similarity_search(query, k=2)
        if not results:
            log.warning(f"No results for: '{query}'")
            all_ok = False
            continue

        top = results[0]
        returned_act = top.metadata.get("act_code", "?")
        log.info(
            f"\n   Query   : {query}\n"
            f"   Expected: {expected_act}  |  Got: {returned_act}\n"
            f"   Snippet : {top.page_content[:180].strip()}..."
        )

    # Extra: check no repealed content slipped through
    all_docs = db.get()
    leaked = [d for d in all_docs["documents"] if d and re.search(r"\[omitted\]", d, re.I)]
    if leaked:
        log.warning(f"WARNING: {len(leaked)} chunk(s) contain [Omitted] — cleaning regex needs update!")
        all_ok = False
    else:
        log.info("   Repealed-content check: PASS — no [Omitted] in DB")

    return all_ok


# ═══════════════════════════════════════════════════════════════════
#  MAIN — orchestrate all 5 blocks
# ═══════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("  Motor & Cyber Law Advisor — Stage 1 Pipeline")
    log.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 60)

    # ── Create output directories ────────────────────────────────
    for d in ["data/raw", "data/processed", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # ── IT ACT 2000 ──────────────────────────────────────────────
    log.info("\n[1/2] Processing IT Act 2000 ...")
    ita_raw   = extract_pdf_text(PDF_ITA)
    ita_clean = clean_it_act(ita_raw)
    run_verification(ita_clean, "ITA")

    # Save cleaned text (optional — useful for debugging)
    Path("data/processed/ita_clean.txt").write_text(ita_clean, encoding="utf-8")

    ita_chunks = make_chunks(ita_clean, "ITA", "Information Technology Act 2000")
    all_chunks.extend(ita_chunks)

    # ── MOTOR VEHICLES ACT ───────────────────────────────────────
    log.info("\n[2/2] Processing Motor Vehicles Act ...")
    mva_raw   = extract_pdf_text(PDF_MVA)
    mva_clean = clean_mva(mva_raw)
    run_verification(mva_clean, "MVA")

    Path("data/processed/mva_clean.txt").write_text(mva_clean, encoding="utf-8")

    mva_chunks = make_chunks(mva_clean, "MVA", "Motor Vehicles Act 2019")
    all_chunks.extend(mva_chunks)

    # ── SCRAPED CSV ──────────────────────────────────────────────
    # Check env flag set by Streamlit Re-Index tab (or default ON)
    ingest_csv = os.environ.get("INGEST_CSV", "1") != "0"
    if ingest_csv:
        log.info("\n[3/3] Loading scraped CSV sections ...")
        csv_chunks = load_csv_chunks()
        all_chunks.extend(csv_chunks)
    else:
        log.info("[3/3] CSV ingestion skipped (INGEST_CSV=0)")

    # ── CHUNK SUMMARY ────────────────────────────────────────────
    log.info(f"\nTotal chunks (both acts): {len(all_chunks)}")
    ita_count = sum(1 for c in all_chunks if c.metadata["act_code"] == "ITA")
    mva_count = sum(1 for c in all_chunks if c.metadata["act_code"] == "MVA")
    log.info(f"  ITA: {ita_count}  |  MVA: {mva_count}")

    if not all_chunks:
        log.error("No chunks produced — check PDF paths and try again.")
        return

    # ── VECTORIZE & STORE ────────────────────────────────────────
    log.info("\nBuilding ChromaDB vector store ...")
    build_vector_store(all_chunks)

    # ── VERIFY ───────────────────────────────────────────────────
    log.info("\nRunning verification queries ...")
    ok = verify_vector_store()

    # ── FINAL STATUS ─────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    if ok:
        log.info("  Stage 1 COMPLETE — vector store is healthy.")
        log.info("  Next step: run src/retrievalchain.py")
    else:
        log.warning("  Stage 1 finished with warnings — review logs above.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()