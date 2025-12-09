"""
Data Validation Script
======================

Validates all preprocessed data files (scraped + transcribed)
for completeness, schema consistency, and quality thresholds.

Checks include:
    - Required fields ('title', 'paragraph')
    - Empty or very short text
    - Duplicate entries
    - Summary statistics

Outputs:
    Logs summary and saves a validation_report.json under /data/validation_reports
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validation")

# ---------------- Project Paths ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_REPORT_DIR = BASE_DATA_DIR / "validation_reports"
os.makedirs(VALIDATION_REPORT_DIR, exist_ok=True)

# ---------------- Helper Functions ---------------- #
def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        logger.warning(f"⚠ File not found: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {path}: {e}")
        return []


def validate_entries(data: List[Dict[str, Any]], source_name: str) -> Dict[str, Any]:
    logger.info(f"Validating {source_name} ({len(data)} records)...")
    total = len(data)
    missing_title = 0
    missing_paragraph = 0
    short_paragraph = 0
    duplicates = 0

    seen_titles = set()
    clean_items = []

    for item in data:
        title = item.get("title", "").strip()
        paragraph = item.get("paragraph", "").strip()

        if not title:
            missing_title += 1
            continue
        if not paragraph:
            missing_paragraph += 1
            continue
        if len(paragraph.split()) < 5:
            short_paragraph += 1
            continue
        if title in seen_titles:
            duplicates += 1
            continue

        seen_titles.add(title)
        clean_items.append(item)

    logger.info(
        f"{source_name} validation summary → "
        f"{len(clean_items)} valid / {total} total | "
        f"Missing title: {missing_title}, Missing paragraph: {missing_paragraph}, "
        f"Short: {short_paragraph}, Duplicates: {duplicates}"
    )

    return {
        "source": source_name,
        "total": total,
        "valid": len(clean_items),
        "missing_title": missing_title,
        "missing_paragraph": missing_paragraph,
        "short_paragraph": short_paragraph,
        "duplicates": duplicates,
    }


def save_report(results: List[Dict[str, Any]]) -> None:
    report_path = VALIDATION_REPORT_DIR / "validation_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Validation report saved → {report_path}")
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")


# ---------------- Main Logic ---------------- #
def main():
    logger.info("Starting validation of preprocessed data...")

    handbook_path = BASE_DATA_DIR / "handbook_paragraphs.json"
    transcripts_path = BASE_DATA_DIR / "meeting_transcripts" / "all_transcripts.json"

    results = []

    handbook_data = load_json(handbook_path)
    if handbook_data:
        results.append(validate_entries(handbook_data, "GitLab Handbook"))

    transcript_data = load_json(transcripts_path)
    if transcript_data:
        results.append(validate_entries(transcript_data, "Meeting Transcripts"))

    if not results:
        logger.warning("⚠ No data files found to validate. Please run scraper/transcription first.")
    else:
        save_report(results)
        logger.info("Validation completed successfully.")


if __name__ == "__main__":
    main()
