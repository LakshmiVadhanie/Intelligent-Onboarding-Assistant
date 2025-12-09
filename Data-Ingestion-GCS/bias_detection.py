"""
Bias Detection Module (Standalone)
==================================
Scans text data (handbook + transcripts) for potential bias or sensitive terms.
Outputs a report highlighting categories and counts.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bias_detection")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = DATA_DIR / "bias_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_PATH = REPORT_DIR / "bias_report.json"

# --- Lexicon-based Bias Detector ---
BIAS_CATEGORIES = {
    "gender": ["he", "she", "man", "woman", "male", "female"],
    "ethnicity": ["asian", "black", "white", "hispanic"],
    "age": ["young", "old", "elderly", "millennial", "senior"],
    "ability": ["disabled", "handicapped"],
    "religion": ["christian", "muslim", "hindu", "jewish"]
}


def detect_bias(text: str) -> Dict[str, int]:
    """Return bias category counts."""
    text_lower = text.lower()
    counts = {cat: 0 for cat in BIAS_CATEGORIES}
    for cat, words in BIAS_CATEGORIES.items():
        for w in words:
            counts[cat] += len(re.findall(rf"\\b{re.escape(w)}\\b", text_lower))
    return counts


def scan_file(file_path: Path) -> Dict[str, Dict[str, int]]:
    """Scan a JSON file and return per-record bias stats."""
    if not file_path.exists():
        logger.warning(f"⚠ File not found: {file_path}")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = {}
    for item in data:
        title = item.get("title", "Untitled")
        paragraph = item.get("paragraph", "")
        text = f"{title} {paragraph}"
        report[title] = detect_bias(text)
    return report


def run_bias_detection():
    logger.info("🚦 Running bias detection...")

    handbook_file = DATA_DIR / "handbook_paragraphs.json"
    transcript_file = DATA_DIR / "meeting_transcripts" / "all_transcripts.json"

    all_reports = {
        "handbook": scan_file(handbook_file),
        "transcripts": scan_file(transcript_file)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    logger.info(f"Bias report saved → {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    run_bias_detection()
