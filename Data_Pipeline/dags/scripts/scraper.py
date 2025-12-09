"""
GitLab Handbook Web Scraper
===========================

This module scrapes structured content from GitLab's public handbook pages:
    https://handbook.gitlab.com/handbook/values/
    https://handbook.gitlab.com/handbook/marketing/blog/
    etc.

It extracts the title and paragraph text and saves them in a structured JSON file.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import json
import os
import re
from urllib.parse import urlsplit
from pathlib import Path
import sys

# ---------------- Logging Setup ---------------- #
# Make sure scripts directory is in sys.path for Airflow
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from logging_utils import get_logger
except ImportError:
    # Fallback for local direct runs
    from dags.scripts.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------- Project Paths ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = PROJECT_ROOT / "data"

# Import preprocessor with dual compatibility
try:
    from preprocess import default_preprocessor
except ImportError:
    sys.path.append(str(CURRENT_DIR))
    from preprocess import default_preprocessor


# ---------------- Core Scraping Logic ---------------- #
def fetch_page_content(url: str) -> str:
    """Fetch raw HTML content from a given URL."""
    logger.info(f"Fetching page content from {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GitLabScraper/1.0; +https://handbook.gitlab.com)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info(f"Fetched content from {url} (size={len(response.text)} bytes)")
        return response.text
    except Exception as e:
        logger.exception(f"Failed to fetch content from {url}: {e}")
        raise


def parse_html_content(html: str) -> Dict[str, Any]:
    """Parse HTML and extract clean paragraph text from the page."""
    logger.info("Parsing HTML content...")
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]

    logger.info(f"Parsed {len(paragraphs)} paragraphs from page: {title}")
    return {"title": title, "paragraphs": paragraphs}


def scrape_gitlab_handbook(url: str) -> Dict[str, Any]:
    """Orchestrates the full scraping pipeline: fetch → parse."""
    logger.info(f"Starting scrape for {url}")
    try:
        html = fetch_page_content(url)
        parsed_data = parse_html_content(html)
        logger.info(f"Scraping complete for {url}")
        return parsed_data
    except Exception as e:
        logger.exception(f"Scraping failed for {url}: {e}")
        raise


# ---------------- Utility Functions ---------------- #
def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")


def safe_filename_from_url(url: str) -> str:
    parts = urlsplit(url)
    base = (parts.netloc + parts.path).rstrip("/") or "index"
    base = base.replace("/", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return f"{base}.json"


def save_json(data: Dict[str, Any], output_path: str) -> None:
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON file → {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save JSON file {output_path}: {e}")
        raise


def default_data_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "data")


# ---------------- CLI Execution ---------------- #
URLS: List[str] = [
    "https://handbook.gitlab.com/handbook/company/mission/",
    "https://handbook.gitlab.com/handbook/values/",
]

if __name__ == "__main__":
    out_dir = default_data_dir()
    ensure_directory(out_dir)

    items: List[Dict[str, str]] = []
    for url in URLS:
        try:
            data = scrape_gitlab_handbook(url)
            pre = default_preprocessor.preprocess_for_scraper(
                data.get("title", ""),
                data.get("paragraphs", [])
            )
            items.append({
                "title": pre["title"],
                "paragraph": " ".join(pre.get("paragraphs", []))
            })
            logger.info(f"Processed {url} with {len(pre.get('paragraphs', []))} cleaned paragraphs.")
        except Exception as e:
            logger.exception(f"Skipping URL due to error: {url} ({e})")

    output_file = os.path.join(out_dir, "handbook_paragraphs.json")
    save_json(items, output_file)
    logger.info(f"Scraper run completed successfully. Output file: {output_file}")