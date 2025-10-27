"""
Data Validation Script
======================

Validates the structure and contents of key JSON outputs:
    - handbook_paragraphs.json
    - meeting_transcripts/all_transcripts.json

If any check fails, exits with code 1 so Airflow detects failure.
"""

import json
import sys
from pathlib import Path

# --- Logging Setup ---
try:
    from .logging_utils import get_logger
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from logging_utils import get_logger

logger = get_logger(__name__)

# --- Base Path ---
BASE_DIR = Path(__file__).resolve().parents[2] / "data"


def validate_json_structure(file_path: Path, required_keys):
    """Validate JSON file structure and contents."""
    if not file_path.exists():
        logger.error(f" Missing file: {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception(f"Error reading {file_path}: {e}")
        return False

    if not isinstance(data, list) or len(data) == 0:
        logger.error(f" {file_path.name}: Empty or invalid list format.")
        return False

    for i, item in enumerate(data[:5]):  # check first few records
        missing = [k for k in required_keys if k not in item]
        if missing:
            logger.error(f"{file_path.name}: Missing keys {missing} in item {i}")
            return False

        for key in required_keys:
            val = item.get(key)
            if val is None or not str(val).strip():
                logger.error(f"{file_path.name}: Empty value for '{key}' in item {i}")
                return False

    logger.info(f" {file_path.name}: Structure and content look valid ({len(data)} records).")
    return True


def run_validation():
    """Run all validation checks."""
    logger.info("Starting data validation checks...")
    handbook_path = BASE_DIR / "handbook_paragraphs.json"
    transcript_path = BASE_DIR / "meeting_transcripts" / "all_transcripts.json"

    checks = [
        validate_json_structure(handbook_path, ["title", "paragraph"]),
        validate_json_structure(transcript_path, ["title", "video_id", "url", "transcript"]),
    ]

    if all(checks):
        logger.info(" All validation checks passed successfully!")
        return True
    else:
        logger.error(" Validation failed — check logs for details.")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
