"""
Bias Detection and Mitigation Module
====================================
Scans text data (handbook + transcripts) for potential bias or sensitive terms.
Outputs a report highlighting categories and severity.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict

# --- Logging Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
from logging_utils import get_logger

logger = get_logger(__name__)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_PATH = BASE_DIR / "bias_report.json"

# --- Simple Lexicon-based Bias Detector ---
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
            counts[cat] += len(re.findall(rf"\b{re.escape(w)}\b", text_lower))
    return counts


def scan_file(file_path: Path) -> Dict[str, Dict[str, int]]:
    """Scan a JSON file and return per-record bias stats."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = {}
    for item in data:
        text = " ".join(str(v) for v in item.values())
        report[item.get("title", "Untitled")] = detect_bias(text)
    return report


def run_bias_detection():
    logger.info(" Running bias detection...")
    handbook_file = BASE_DIR / "handbook_paragraphs.json"
    transcript_file = BASE_DIR / "meeting_transcripts" / "all_transcripts.json"

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