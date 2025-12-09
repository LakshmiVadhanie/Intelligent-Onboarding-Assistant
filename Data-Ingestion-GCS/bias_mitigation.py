"""
Bias Mitigation Module (Standalone)
===================================
Uses bias_report.json to mitigate biased language in the text data by:
    - Replacing gendered or sensitive terms with neutral ones.
    - Creating "debiased" versions of the original JSON files.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bias_mitigation")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_FILE = DATA_DIR / "bias_reports" / "bias_report.json"
OUTPUT_DIR = DATA_DIR / "debiased_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Replacement Dictionary ---
REPLACEMENTS: Dict[str, str] = {
    "he": "they", "she": "they",
    "him": "them", "her": "them",
    "his": "their", "hers": "their",
    "man": "person", "woman": "person",
    "mankind": "humankind", "chairman": "chairperson",
    "old": "experienced", "young": "early-career",
    "disabled": "person with disability",
}


def debias_text(text: str) -> str:
    """Replace biased terms with neutral alternatives."""
    words = text.split()
    return " ".join([REPLACEMENTS.get(w.lower(), w) for w in words])


def process_file(input_path: Path, output_path: Path, fields: List[str]):
    """Debias the given JSON file field-wise."""
    if not input_path.exists():
        logger.warning(f"⚠ File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        for field in fields:
            if field in item and isinstance(item[field], str):
                item[field] = debias_text(item[field])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Debiased file saved → {output_path}")


def run_bias_mitigation():
    """Run mitigation after bias detection."""
    logger.info("🧹 Running bias mitigation...")

    handbook_path = DATA_DIR / "handbook_paragraphs.json"
    transcripts_path = DATA_DIR / "meeting_transcripts" / "all_transcripts.json"

    debiased_handbook = OUTPUT_DIR / "handbook_paragraphs_debiased.json"
    debiased_transcripts = OUTPUT_DIR / "all_transcripts_debiased.json"

    process_file(handbook_path, debiased_handbook, ["title", "paragraph"])
    process_file(transcripts_path, debiased_transcripts, ["title", "paragraph"])

    logger.info("Bias mitigation complete. Debiased data ready for ingestion.")
    return OUTPUT_DIR


if __name__ == "__main__":
    run_bias_mitigation()