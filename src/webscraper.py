"""
╔══════════════════════════════════════════════════════════════════╗
║     MOTOR & CYBER LAW ADVISOR — STAGE 1 WEB SCRAPER             ║
║     Scrapes 3 Indian Legal Websites → Saves structured CSVs     ║
╚══════════════════════════════════════════════════════════════════╝

Websites covered:
  1. India Code (indiacode.nic.in)  → IT Act 2000 + Motor Vehicles Act ToC
  2. MoRTH      (morth.nic.in)      → Motor Transport notifications
  3. MeitY      (meity.gov.in)      → IT Act notifications & circulars

Output CSVs (all saved to data/raw/scraped/):
  - indiacode_it_act_toc.csv
  - indiacode_mva_toc.csv
  - morth_notifications.csv
  - meity_notifications.csv
  - combined_legal_toc.csv  ← merged, cleaned, ready for Stage 1 chunking

Install:
  pip install requests beautifulsoup4 pandas lxml

Run:
  python stage1_web_scraper.py
"""

# ─────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────
import re
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("data/raw/scraped")
REQUEST_DELAY  = 2.0          # seconds between requests — be polite
REQUEST_TIMEOUT = 20
MAX_RETRIES    = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── Target URLs ─────────────────────────────────────────────────
URLS = {
    "india_code_ita": "https://indiacode.nic.in/handle/123456789/1999",
    "india_code_mva": "https://indiacode.nic.in/handle/123456789/13960",
    "morth_notifications": "https://morth.nic.in/notification",
    "morth_circulars":     "https://morth.nic.in/circular",
    "meity_acts":          "https://www.meity.gov.in/acts-rules",
    "meity_notifications": "https://www.meity.gov.in/notifications",
}

# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/scraper_{datetime.now():%Y%m%d_%H%M%S}.log"
        ),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 1 — HTTP FETCHER
#  Single responsibility: fetch raw HTML with retry + back-off
# ═══════════════════════════════════════════════════════════════════
def fetch_html(url: str, label: str = "") -> str | None:
    """
    Fetch HTML from a URL with exponential back-off retry.

    Args:
        url   : Target URL
        label : Human-readable name for log messages

    Returns:
        HTML string on success, None on permanent failure.
    """
    tag = label or url[:60]
    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"Fetching [{tag}] attempt {attempt}/{MAX_RETRIES}")
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            log.info(f"  OK — {len(resp.text):,} chars | status {resp.status_code}")
            return resp.text

        except requests.exceptions.HTTPError as e:
            # 403/404/410 — no point retrying
            log.warning(f"  HTTP {e.response.status_code} — skipping")
            return None

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = attempt * 3
            log.warning(f"  Connection error: {e} — retrying in {wait}s")
            time.sleep(wait)

        except Exception as e:
            log.error(f"  Unexpected error: {e}")
            return None

    log.error(f"All {MAX_RETRIES} attempts failed for: {url}")
    return None


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 2 — INDIA CODE PARSER
#  Scrapes Table of Contents from indiacode.nic.in
#
#  India Code page structure:
#   <div class="simple-item-view-description">
#     <div class="word-break">
#       Section 1. Short title...
#       Section 2. Definitions...
#  OR:
#   <div class="toc"> <ul> <li><a href="/handle/...">Section...</a></li>
# ═══════════════════════════════════════════════════════════════════
def parse_india_code_toc(html: str, act_code: str, act_name: str,
                          base_url: str = "https://indiacode.nic.in") -> list[dict]:
    """
    Parse Table of Contents from India Code portal.

    Tries multiple CSS selectors in priority order because India Code
    uses different page templates for different acts.

    Returns:
        List of dicts: {act_code, act_name, section_number,
                        section_title, url, is_repealed, chapter, scraped_at}
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    # ── Strategy 1: explicit TOC div with <li> links ──────────────
    toc = (
        soup.find("div", class_="toc") or
        soup.find("div", class_="simple-item-view-description") or
        soup.find("div", id="content-main") or
        soup.find("main")
    )

    if not toc:
        log.warning(f"  No TOC container found for {act_code} — using body fallback")
        toc = soup.body

    # ── Strategy 2: find all section-like <a> tags ─────────────────
    links = toc.find_all("a", href=True) if toc else []
    section_links = [
        l for l in links
        if "/handle/" in l.get("href", "") or
        re.search(r"section|chapter|schedule", l.get_text(), re.I)
    ]

    if section_links:
        log.info(f"  Strategy 1 (links): found {len(section_links)} items")
        current_chapter = ""
        for link in section_links:
            raw_text = _clean_text(link.get_text())
            if not raw_text or len(raw_text) < 3:
                continue

            # Detect chapter headings
            if re.match(r"^CHAPTER\s+[IVXLC0-9]+", raw_text, re.I):
                current_chapter = raw_text
                continue

            sec_num = _extract_section_number(raw_text)
            href = link.get("href", "")
            full_url = base_url + href if href.startswith("/") else href

            records.append({
                "act_code":       act_code,
                "act_name":       act_name,
                "chapter":        current_chapter,
                "section_number": sec_num,
                "section_title":  raw_text,
                "url":            full_url,
                "is_repealed":    _is_repealed(raw_text),
                "scraped_at":     datetime.utcnow().isoformat(),
                "source":         "indiacode",
            })
        return records

    # ── Strategy 3: parse raw text lines from description div ──────
    log.info(f"  Strategy 2 (text lines): parsing raw text for {act_code}")
    if toc:
        raw = toc.get_text(separator="\n")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        current_chapter = ""
        for line in lines:
            if len(line) < 4:
                continue
            if re.match(r"^CHAPTER\s+[IVXLC0-9]+", line, re.I):
                current_chapter = line
                continue
            if not re.search(r"^\d|^Section|^Sec\.", line, re.I):
                continue
            sec_num = _extract_section_number(line)
            records.append({
                "act_code":       act_code,
                "act_name":       act_name,
                "chapter":        current_chapter,
                "section_number": sec_num,
                "section_title":  _clean_text(line),
                "url":            "",
                "is_repealed":    _is_repealed(line),
                "scraped_at":     datetime.utcnow().isoformat(),
                "source":         "indiacode",
            })

    log.info(f"  Parsed {len(records)} records for {act_code}")
    return records


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 3 — MORTH NOTIFICATIONS PARSER
#  Scrapes notifications list from morth.nic.in
#
#  MoRTH page structure (Drupal views):
#   <div class="view-content">
#     <div class="views-row">
#       <span class="date-display-single">15 Jan 2024</span>
#       <a href="/sites/.../notification.pdf">Title text</a>
#     </div>
# ═══════════════════════════════════════════════════════════════════
def parse_morth(html: str, category: str = "notification") -> list[dict]:
    """
    Parse MoRTH notification/circular listings.

    Args:
        html     : Raw HTML from morth.nic.in
        category : "notification" or "circular"

    Returns:
        List of dicts: {title, date, pdf_url, category, scraped_at}
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    # MoRTH uses Drupal — primary container is .view-content
    container = (
        soup.find("div", class_="view-content") or
        soup.find("div", class_="field-items") or
        soup.find("div", id="content") or
        soup.body or
        soup
    )

    # Each item is a .views-row or <li> or <tr>
    items = (
        container.find_all("div", class_=re.compile(r"views-row")) or
        container.find_all("li") or
        container.find_all("tr")
    )

    if not items:
        # Fallback: any link with .pdf extension
        items = soup.find_all("a", href=re.compile(r"\.pdf", re.I))
        log.info(f"  MoRTH fallback: found {len(items)} PDF links")
        for link in items:
            title = _clean_text(link.get_text())
            href  = link.get("href", "")
            records.append({
                "act_code":   "MVA",
                "act_name":   "Motor Vehicles Act",
                "title":      title or "Untitled",
                "date":       "",
                "pdf_url":    _absolute_url(href, "https://morth.nic.in"),
                "category":   category,
                "scraped_at": datetime.utcnow().isoformat(),
                "source":     "morth",
            })
        return records

    log.info(f"  MoRTH: found {len(items)} items")
    for item in items:
        # Extract date
        date_tag = (
            item.find("span", class_=re.compile(r"date")) or
            item.find("span", class_="date-display-single") or
            item.find("time")
        )
        date_str = _clean_text(date_tag.get_text()) if date_tag else ""

        # Extract link + title
        link = item.find("a", href=True)
        if not link:
            continue
        title    = _clean_text(link.get_text())
        href     = link.get("href", "")
        pdf_url  = _absolute_url(href, "https://morth.nic.in")

        if not title or len(title) < 5:
            continue

        records.append({
            "act_code":   "MVA",
            "act_name":   "Motor Vehicles Act",
            "title":      title,
            "date":       date_str,
            "pdf_url":    pdf_url,
            "category":   category,
            "scraped_at": datetime.utcnow().isoformat(),
            "source":     "morth",
        })

    log.info(f"  Parsed {len(records)} MoRTH records")
    return records


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 4 — MEITY PARSER
#  Scrapes IT Act notifications and acts from meity.gov.in
#
#  MeitY page structure:
#   <div class="view-content">
#     <div class="views-row">
#       <div class="views-field-title">
#         <span class="field-content"><a href="...">Title</a>
#       </div>
#       <div class="views-field-field-date"> ... </div>
# ═══════════════════════════════════════════════════════════════════
def parse_meity(html: str, category: str = "notification") -> list[dict]:
    """
    Parse MeitY acts/notifications page.

    Returns:
        List of dicts: {title, date, url, category, scraped_at}
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    # MeitY uses Drupal views similar to MoRTH
    container = (
        soup.find("div", class_="view-content") or
        soup.find("div", class_="content") or
        soup.find("main") or
        soup.body or
        soup
    )

    items = (
        container.find_all("div", class_=re.compile(r"views-row")) or
        container.find_all("li") or
        []
    )

    if not items:
        # Fallback: all links in the content area
        all_links = container.find_all("a", href=True) if container else []
        relevant  = [
            l for l in all_links
            if len(l.get_text(strip=True)) > 10
        ]
        log.info(f"  MeitY fallback: {len(relevant)} content links")
        for link in relevant:
            title = _clean_text(link.get_text())
            href  = link.get("href", "")
            records.append({
                "act_code":   "ITA",
                "act_name":   "IT Act 2000",
                "title":      title,
                "date":       "",
                "url":        _absolute_url(href, "https://www.meity.gov.in"),
                "category":   category,
                "scraped_at": datetime.utcnow().isoformat(),
                "source":     "meity",
            })
        return records

    log.info(f"  MeitY: found {len(items)} items")
    for item in items:
        # Title link
        title_div = (
            item.find("div", class_=re.compile(r"title")) or
            item.find("span", class_=re.compile(r"title"))
        )
        link = (title_div.find("a", href=True) if title_div
                else item.find("a", href=True))
        if not link:
            continue

        title = _clean_text(link.get_text())
        href  = link.get("href", "")

        # Date
        date_div = item.find("div", class_=re.compile(r"date|field-date"))
        date_str = _clean_text(date_div.get_text()) if date_div else ""

        if not title or len(title) < 5:
            continue

        records.append({
            "act_code":   "ITA",
            "act_name":   "IT Act 2000",
            "title":      title,
            "date":       date_str,
            "url":        _absolute_url(href, "https://www.meity.gov.in"),
            "category":   category,
            "scraped_at": datetime.utcnow().isoformat(),
            "source":     "meity",
        })

    log.info(f"  Parsed {len(records)} MeitY records")
    return records


# ═══════════════════════════════════════════════════════════════════
#  BLOCK 5 — OUTPUT WRITER
#  Saves per-source CSVs + one combined master CSV
# ═══════════════════════════════════════════════════════════════════
def save_csv(records: list[dict], filename: str) -> pd.DataFrame:
    """
    Save records to CSV and return as DataFrame.
    Skips save if records is empty.
    """
    if not records:
        log.warning(f"  No records to save for {filename}")
        return pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"  Saved {len(df)} rows → {path}")
    return df


def save_combined(all_frames: list[pd.DataFrame]) -> None:
    """Merge all scraped DataFrames into one master CSV."""
    non_empty = [df for df in all_frames if not df.empty]
    if not non_empty:
        log.warning("No data collected — combined CSV not written")
        return

    combined = pd.concat(non_empty, ignore_index=True)

    # Standardise column names across sources
    combined = combined.fillna("")

    # Flag repealed entries for filtering in Stage 1
    if "section_title" in combined.columns:
        combined["is_repealed"] = combined["section_title"].apply(
            lambda t: bool(re.search(r"\[Omitted\]|\[Rep\.", str(t), re.I))
        )
    if "title" in combined.columns and "is_repealed" not in combined.columns:
        combined["is_repealed"] = False

    path = OUTPUT_DIR / "combined_legal_toc.csv"
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"\nCombined CSV: {len(combined)} total rows → {path}")

    # Summary by source
    if "source" in combined.columns:
        summary = combined.groupby("source").size().reset_index(name="count")
        log.info("\nSummary by source:\n" + summary.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def _clean_text(raw: str) -> str:
    """Strip whitespace and normalise dashes/quotes."""
    text = re.sub(r"\s+", " ", raw.strip())
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def _extract_section_number(text: str) -> str:
    """
    Pull out the section number from text like:
      '43A. Compensation for failure...'
      'Section 66F. Punishment for cyber terrorism.'
      '3. Authentication of electronic records.'
    """
    match = re.search(
        r"(?:Section|Sec\.?|§)?\s*(\d+[A-Za-z]?)[.\s]",
        text,
        re.IGNORECASE
    )
    return match.group(1) if match else ""


def _is_repealed(text: str) -> bool:
    """Return True if text indicates a repealed/omitted section."""
    return bool(re.search(
        r"\[Omitted\]|\[Rep\.\]|\[Repealed|Omitted by",
        text, re.IGNORECASE
    ))


def _absolute_url(href: str, base: str) -> str:
    """Convert relative /path to absolute https://base/path."""
    href = href.strip()
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return base.rstrip("/") + href
    return href


# ═══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("  Motor & Cyber Law Advisor — Stage 1 Web Scraper")
    log.info(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 65)

    all_frames = []

    # ── 1. India Code — IT Act 2000 ToC ─────────────────────────
    log.info("\n[1/4] India Code — IT Act 2000")
    html = fetch_html(URLS["india_code_ita"], "IndiaCode IT Act")
    if html:
        records = parse_india_code_toc(
            html, "ITA", "Information Technology Act 2000"
        )
        df = save_csv(records, "indiacode_it_act_toc.csv")
        all_frames.append(df)
    time.sleep(REQUEST_DELAY)

    # ── 2. India Code — Motor Vehicles Act ToC ───────────────────
    log.info("\n[2/4] India Code — Motor Vehicles Act")
    html = fetch_html(URLS["india_code_mva"], "IndiaCode MVA")
    if html:
        records = parse_india_code_toc(
            html, "MVA", "Motor Vehicles Act 2019"
        )
        df = save_csv(records, "indiacode_mva_toc.csv")
        all_frames.append(df)
    time.sleep(REQUEST_DELAY)

    # ── 3. MoRTH — Notifications ─────────────────────────────────
    log.info("\n[3/4] MoRTH — Notifications")
    html = fetch_html(URLS["morth_notifications"], "MoRTH Notifications")
    if html:
        records = parse_morth(html, category="notification")
        df = save_csv(records, "morth_notifications.csv")
        all_frames.append(df)
    time.sleep(REQUEST_DELAY)

    # ── 4. MeitY — IT Act Notifications ──────────────────────────
    log.info("\n[4/4] MeitY — Notifications")
    html = fetch_html(URLS["meity_notifications"], "MeitY Notifications")
    if html:
        records = parse_meity(html, category="notification")
        df = save_csv(records, "meity_notifications.csv")
        all_frames.append(df)

    # ── 5. Combine & save master CSV ─────────────────────────────
    log.info("\nMerging all sources...")
    save_combined(all_frames)

    log.info("\n" + "=" * 65)
    log.info("  Scraping complete.")
    log.info(f"  Output directory: {OUTPUT_DIR.resolve()}")
    log.info("  Next step: run stage1_ingestion.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()