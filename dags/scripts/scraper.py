"""
GitLab Handbook Web Scraper
===========================

This module provides functions to scrape structured content from
GitLab's public handbook pages such as:
    https://handbook.gitlab.com/handbook/values/
    https://handbook.gitlab.com/handbook/marketing/blog/
    ...

It extracts the main textual content — including the page title, headings,
and paragraphs — and returns a structured dictionary object.

Dependencies:
    - requests
    - beautifulsoup4

Install using:
    pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import json
import os
import re
from urllib.parse import urlsplit
from pathlib import Path

# ---------------- Logging Setup ---------------- #
try:
    from .logging_utils import get_logger
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from logging_utils import get_logger

logger = get_logger(__name__)

# ---------------- Project Paths ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = PROJECT_ROOT / "data"

try:
    # Works when run as part of a package (python -m ...)
    from .preprocess import default_preprocessor
except ImportError:
    # Works when run directly (python dags/scripts/scraper.py)
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from preprocess import default_preprocessor


def fetch_page_content(url: str) -> str:
    """
    Fetch raw HTML content from a given URL.
    """
    logger.info(f"Fetching page content from {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GitLabScraper/1.0; +https://handbook.gitlab.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        logger.info(f"Successfully fetched content from {url} (size={len(response.text)} bytes)")
        return response.text
    except Exception as e:
        logger.exception(f"Failed to fetch content from {url}: {e}")
        raise


def parse_html_content(html: str) -> Dict[str, Any]:
    """
    Parse the HTML and extract only clean paragraph text from the page.
    """
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
    """
    Orchestrates the full scraping pipeline: fetch → parse → return only paragraphs.
    """
    logger.info(f"Starting scrape for {url}")
    try:
        html = fetch_page_content(url)
        parsed_data = parse_html_content(html)
        logger.info(f"Scraping complete for {url}")
        return parsed_data
    except Exception as e:
        logger.exception(f"Scraping failed for {url}: {e}")
        raise


# ---------------------- Utilities ---------------------- #
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
    from pathlib import Path
    return str(Path(__file__).resolve().parents[2] / "data")


# ---------------------- CLI Execution ---------------------- #
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
            logger.info(f"Processed {url} successfully with {len(pre.get('paragraphs', []))} cleaned paragraphs.")
        except Exception as e:
            logger.exception(f"Skipping URL due to error: {url} ({e})")

    output_file = os.path.join(out_dir, "handbook_paragraphs.json")
    save_json(items, output_file)
    logger.info(f"✅ Scraper run completed. Output file: {output_file}")
