"""
╔══════════════════════════════════════════════════════════════════╗
║      MOTOR & CYBER LAW ADVISOR — EVALUATION HARNESS             ║
║      Runs 50+ realistic user questions and logs all results      ║
╚══════════════════════════════════════════════════════════════════╝

Purpose:
  Stress-test the RAG pipeline with a wide variety of real-world
  user questions across ALL major topic areas of the two acts:

    Motor Vehicles Act 2019 (MVA) — traffic offences, licences,
        insurance, accidents, permits, vehicle registration, etc.

    Information Technology Act 2000 (ITA) — cyber crimes, hacking,
        identity theft, data protection, Section 66A, e-commerce, etc.

  Results are printed to the console AND written to:
    data/eval_results.csv   ← for Excel / Power BI analysis

Run:
    python src/evaluate.py
"""

# ─────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Reuse all pipeline components from the main retrieval chain
from retrievalchain import (
    load_vector_store,
    build_llm,
    init_db,
    rag_query,
    DISCLAIMER,
)

# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,   # suppress INFO noise during bulk eval
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  TEST SUITE
#  Format: (question_text, expected_act_or_None, category_label)
#
#  expected_act:
#    "MVA"  → expect a cited answer about Motor Vehicles Act
#    "ITA"  → expect a cited answer about IT Act
#    None   → expect the DISCLAIMER (out-of-scope question)
# ═══════════════════════════════════════════════════════════════════
TEST_SUITE = [

    # ── MOTOR VEHICLES ACT — Driving Licence ─────────────────────
    ("Can I drive a car without a driving licence?",                     "MVA", "MVA_Licence"),
    ("What is the minimum age to apply for a driving licence in India?", "MVA", "MVA_Licence"),
    ("What documents are required to get a driving licence?",            "MVA", "MVA_Licence"),
    ("What happens if my driving licence expires?",                      "MVA", "MVA_Licence"),
    ("Can a learner's licence holder drive alone on the road?",          "MVA", "MVA_Licence"),
    ("What is the validity period of a driving licence?",                "MVA", "MVA_Licence"),
    ("Can my driving licence be suspended?",                             "MVA", "MVA_Licence"),

    # ── MOTOR VEHICLES ACT — Penalties & Offences ────────────────
    ("What is the fine for drunk driving in India?",                     "MVA", "MVA_Penalties"),
    ("What is the penalty for driving without insurance?",               "MVA", "MVA_Penalties"),
    ("What is the fine for jumping a red traffic signal?",               "MVA", "MVA_Penalties"),
    ("What is the penalty for driving without wearing a seatbelt?",      "MVA", "MVA_Penalties"),
    ("What is the fine for not wearing a helmet while riding a bike?",   "MVA", "MVA_Penalties"),
    ("What happens if I use a mobile phone while driving?",              "MVA", "MVA_Penalties"),
    ("What is the punishment for racing on public roads?",               "MVA", "MVA_Penalties"),
    ("What is the penalty for overloading a vehicle?",                   "MVA", "MVA_Penalties"),
    ("What is the fine for driving without a valid registration?",       "MVA", "MVA_Penalties"),
    ("What is the punishment for driving without a permit?",             "MVA", "MVA_Penalties"),

    # ── MOTOR VEHICLES ACT — Vehicle Registration & Insurance ────
    ("Is vehicle insurance mandatory in India?",                         "MVA", "MVA_Insurance"),
    ("What is a third-party insurance policy for vehicles?",             "MVA", "MVA_Insurance"),
    ("Can I drive an unregistered vehicle in India?",                    "MVA", "MVA_Registration"),
    ("What is the process for vehicle registration?",                    "MVA", "MVA_Registration"),
    ("What is the penalty for using a vehicle with a fake number plate?","MVA", "MVA_Registration"),

    # ── MOTOR VEHICLES ACT — Accidents & Compensation ────────────
    ("What should I do after a road accident in India?",                 "MVA", "MVA_Accident"),
    ("Can I get compensation if I'm injured in a road accident?",        "MVA", "MVA_Accident"),
    ("What is the motor accident claims tribunal?",                      "MVA", "MVA_Accident"),
    ("Is a hit-and-run accident driver punishable under MVA?",           "MVA", "MVA_Accident"),
    ("Who is liable to pay compensation in a road accident?",            "MVA", "MVA_Accident"),

    # ── MOTOR VEHICLES ACT — Miscellaneous ───────────────────────
    ("What is the speed limit for vehicles on highways?",                "MVA", "MVA_Misc"),
    ("Can a minor drive a vehicle with parental permission?",            "MVA", "MVA_Misc"),
    ("What are the rules for wearing a seatbelt in rear seats?",         "MVA", "MVA_Misc"),
    ("Can police seize my vehicle for traffic violations?",              "MVA", "MVA_Misc"),

    # ── IT ACT — Cyber Crimes & Punishments ──────────────────────
    ("What is the punishment for hacking a computer system?",            "ITA", "ITA_CyberCrime"),
    ("What is the penalty for identity theft under IT Act?",             "ITA", "ITA_CyberCrime"),
    ("What is the punishment for sending obscene content online?",       "ITA", "ITA_CyberCrime"),
    ("Is phishing illegal in India? What is the punishment?",            "ITA", "ITA_CyberCrime"),
    ("What is the punishment for cyber stalking in India?",              "ITA", "ITA_CyberCrime"),
    ("What is the punishment for cheating using computers?",             "ITA", "ITA_CyberCrime"),
    ("Is spreading malware or viruses a crime under Indian law?",        "ITA", "ITA_CyberCrime"),
    ("What is the punishment for stealing someone's password?",          "ITA", "ITA_CyberCrime"),
    ("What is data theft under the IT Act?",                             "ITA", "ITA_CyberCrime"),
    ("What is the punishment for cyber terrorism in India?",             "ITA", "ITA_CyberCrime"),

    # ── IT ACT — Electronic Signatures & Evidence ────────────────
    ("Is an electronic signature legally valid in India?",               "ITA", "ITA_Esign"),
    ("What is a digital signature certificate?",                         "ITA", "ITA_Esign"),
    ("Are electronic contracts legally enforceable in India?",           "ITA", "ITA_Esign"),
    ("What is a certifying authority under the IT Act?",                 "ITA", "ITA_Esign"),

    # ── IT ACT — Data Protection & Intermediaries ────────────────
    ("What are the responsibilities of internet intermediaries?",        "ITA", "ITA_Intermediary"),
    ("Can a social media platform be held liable for user content?",     "ITA", "ITA_Intermediary"),
    ("What is the safe harbour protection for intermediaries?",          "ITA", "ITA_Intermediary"),
    ("What obligations does a company have to protect user data?",       "ITA", "ITA_DataProt"),

    # ── IT ACT — Section 66A (Special Case) ──────────────────────
    ("Can I be arrested for posting offensive content online?",          "ITA", "ITA_66A"),
    ("Is Section 66A of the IT Act still valid law?",                    "ITA", "ITA_66A"),

    # ── OUT-OF-SCOPE (should return DISCLAIMER) ───────────────────
    ("What are the rules for drone registration in India?",              None,  "OOS"),
    ("What is the GST rate on electric vehicles?",                       None,  "OOS"),
    ("What are the labour laws for gig workers in India?",               None,  "OOS"),
    ("What is the penalty for income tax evasion in India?",             None,  "OOS"),
    ("What is the right to privacy under the Indian Constitution?",      None,  "OOS"),
    ("What are the gun licence rules in India?",                         None,  "OOS"),
]


# ─────────────────────────────────────────────────────────────────
#  RATE-LIMIT CONFIGURATION
#  Groq has strict tokens-per-minute rate limits.
#  Set to >= 13s to stay within limits. Paid-tier users: set to 0.
# ─────────────────────────────────────────────────────────────────
RATE_LIMIT_DELAY_SEC = 2   # seconds to wait between Groq calls


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_evaluation(db, llm, conn) -> dict:
    """
    Run all test queries and collect results.

    Returns a summary dict with pass/fail counts and category breakdown.
    """
    total  = len(TEST_SUITE)
    passed = 0
    failed = 0
    errors = 0

    category_stats: dict[str, dict] = {}
    rows: list[dict] = []

    print("\n" + "═" * 70)
    print(f"  LEGAL RAG EVALUATION — {total} TEST QUERIES")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Rate-limit delay: {RATE_LIMIT_DELAY_SEC}s between queries")
    print("═" * 70)

    for i, (question, expected_act, category) in enumerate(TEST_SUITE, 1):
        if category not in category_stats:
            category_stats[category] = {"pass": 0, "fail": 0, "total": 0}
        category_stats[category]["total"] += 1

        print(f"\n[{i:02d}/{total}] [{category}] {question}")

        # ── Rate-limit pause ──────────────────────────────────────
        if i > 1 and RATE_LIMIT_DELAY_SEC > 0:
            print(f"   ⏳ Waiting {RATE_LIMIT_DELAY_SEC}s (rate limit)...")
            time.sleep(RATE_LIMIT_DELAY_SEC)

        try:
            result = rag_query(db, llm, conn, question)
            answer       = result["answer"]
            answer_found = result["answer_found"]
            latency_ms   = result["latency_ms"]
            top_act      = result["chunks"][0].metadata.get("act_code", "?") if result["chunks"] else "—"

            # Determine PASS / FAIL
            if expected_act is None:
                # We expect a disclaimer
                status = "PASS" if answer_found == 0 else "FAIL"
                note   = "Correctly returned disclaimer" if status == "PASS" else "Expected disclaimer but got cited answer"
            else:
                # We expect a cited answer
                status = "PASS" if answer_found == 1 else "FAIL"
                note   = "Correctly cited an answer" if status == "PASS" else "Expected cited answer but got disclaimer"

            if status == "PASS":
                passed += 1
                category_stats[category]["pass"] += 1
                print(f"   ✓ PASS  | Latency: {latency_ms}ms | Top chunk: {top_act}")
            else:
                failed += 1
                category_stats[category]["fail"] += 1
                print(f"   ✗ FAIL  | Latency: {latency_ms}ms | {note}")
                print(f"   Answer preview: {answer[:200]}...")

        except Exception as exc:
            errors += 1
            status, note, latency_ms, top_act = "ERROR", str(exc), 0, "—"
            print(f"   ✗ ERROR | {exc}")

        rows.append({
            "index":        i,
            "category":     category,
            "question":     question,
            "expected_act": expected_act or "OOS",
            "status":       status,
            "latency_ms":   latency_ms,
            "top_act":      top_act,
            "note":         note,
        })

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  RESULTS SUMMARY")
    print("═" * 70)
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}  ({100*passed/total:.1f}%)")
    print(f"  Failed : {failed}  ({100*failed/total:.1f}%)")
    if errors:
        print(f"  Errors : {errors}")

    print("\n  Results by Category:")
    print(f"  {'Category':<25} {'Pass':>5} {'Fail':>5} {'Total':>6} {'Rate':>7}")
    print("  " + "─" * 50)
    for cat, stats in sorted(category_stats.items()):
        rate = 100 * stats["pass"] / stats["total"] if stats["total"] else 0
        print(f"  {cat:<25} {stats['pass']:>5} {stats['fail']:>5} {stats['total']:>6} {rate:>6.0f}%")
    print("═" * 70 + "\n")

    return {
        "rows":     rows,
        "total":    total,
        "passed":   passed,
        "failed":   failed,
        "errors":   errors,
        "category_stats": category_stats,
    }


# ═══════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════
def export_csv(rows: list[dict], path: str = "data/eval_results.csv") -> None:
    """Save evaluation rows to a CSV file for Power BI / Excel analysis."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "category", "question", "expected_act",
                  "status", "latency_ms", "top_act", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Evaluation results saved to: {path}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("Loading pipeline components...")
    db   = load_vector_store()
    llm  = build_llm()
    conn = init_db()

    summary = run_evaluation(db, llm, conn)
    export_csv(summary["rows"])

    conn.close()
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
