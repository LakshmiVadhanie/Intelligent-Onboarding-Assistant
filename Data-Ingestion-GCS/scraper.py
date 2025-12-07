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
import logging

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scraper")

# ---------------- Project Paths ---------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_ROOT / "data"

# Import preprocessor (works locally)
try:
    from preprocess import default_preprocessor
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from preprocess import default_preprocessor


# ---------------- Core Scraping Logic ---------------- #
def fetch_page_content(url: str) -> str:
    """Fetch raw HTML content from a given URL."""
    logger.info(f"Fetching page content from {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GitLabScraper/1.0; +https://handbook.gitlab.com)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        logger.info(f"Fetched content from {url} (size={len(response.text)} bytes)")
        return response.text
    except Exception as e:
        logger.error(f"Failed to fetch content from {url}: {e}")
        return ""


def parse_html_content(html: str) -> Dict[str, Any]:
    """Parse HTML and extract clean paragraph text from the page."""
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
    html = fetch_page_content(url)
    if not html:
        return {"title": None, "paragraphs": []}
    parsed_data = parse_html_content(html)
    return parsed_data


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
        logger.error(f"Failed to save JSON file {output_path}: {e}")


def default_data_dir() -> str:
    return str(Path(__file__).resolve().parent / "data")


# ---------------- CLI Execution ---------------- #
URLS: List[str] = [
    # "https://about.gitlab.com/company/",
    # "https://handbook.gitlab.com/handbook/company/mission/",
    # "https://handbook.gitlab.com/handbook/values/",
    # "https://handbook.gitlab.com/handbook/communication/",
    # "https://handbook.gitlab.com/handbook/company/culture/",
    # "https://handbook.gitlab.com/teamops/",
    # "https://handbook.gitlab.com/handbook/company/e-group-weekly/",
    # "https://handbook.gitlab.com/handbook/company/esg/",
    # "https://handbook.gitlab.com/handbook/ceo/office-of-the-ceo/",
    # "https://handbook.gitlab.com/handbook/people-group/anti-harassment/",
    # "https://handbook.gitlab.com/handbook/hiring/",
    # "https://handbook.gitlab.com/handbook/company/culture/inclusion/",
    # "https://handbook.gitlab.com/handbook/leadership/",
    # "https://handbook.gitlab.com/handbook/people-group/learning-and-development/",
    # "https://handbook.gitlab.com/handbook/people-group/general-onboarding/",
    # "https://handbook.gitlab.com/handbook/people-group/offboarding/",
    # "https://handbook.gitlab.com/handbook/finance/spending-company-money/",
    # "https://handbook.gitlab.com/handbook/people-group/talent-assessment/",
    # "https://handbook.gitlab.com/handbook/people-group/team-member-relations/#team-member-relations-philosophy",
    # "https://handbook.gitlab.com/handbook/total-rewards/",
    # "https://handbook.gitlab.com/handbook/tools-and-tips/",
    # "https://handbook.gitlab.com/handbook/support/",
    # "https://handbook.gitlab.com/handbook/engineering/development/",
    # "https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/",
    # "https://handbook.gitlab.com/handbook/security/",
    # "https://handbook.gitlab.com/handbook/engineering/open-source/",
    # "https://handbook.gitlab.com/handbook/security/policies_and_standards/",
    # "https://handbook.gitlab.com/handbook/security/product-security/",
    # "https://handbook.gitlab.com/handbook/security/security-operations/",
    # "https://handbook.gitlab.com/handbook/security/threat-management/",
    # "https://handbook.gitlab.com/handbook/security/security-assurance/",
    # "https://handbook.gitlab.com/handbook/finance/accounts-payable/",
    # "https://handbook.gitlab.com/handbook/finance/accounting/#accounts-receivable",
    # "https://handbook.gitlab.com/handbook/business-technology/",
    # "https://handbook.gitlab.com/handbook/finance/expenses/",
    # "https://handbook.gitlab.com/handbook/finance/financial-planning-and-analysis/",
    # "https://handbook.gitlab.com/handbook/finance/payroll/",
    # "https://handbook.gitlab.com/handbook/finance/procurement/",
    # "https://handbook.gitlab.com/handbook/finance/tax/",
    # "https://handbook.gitlab.com/handbook/board-meetings/",
    # "https://handbook.gitlab.com/handbook/finance/internal-audit/",
    # "https://handbook.gitlab.com/handbook/total-rewards/stock-options/"
    # "https://handbook.gitlab.com/handbook/marketing/blog/release-posts/",
    # "https://handbook.gitlab.com/handbook/product/categories/gitlab-the-product/",
    # "https://handbook.gitlab.com/handbook/product/product-management/",
    # "https://handbook.gitlab.com/handbook/product/product-principles/",
    # "https://handbook.gitlab.com/handbook/product/product-processes/",
    # "https://handbook.gitlab.com/handbook/product-development/how-we-work/product-development-flow/",
    # "https://handbook.gitlab.com/handbook/engineering/workflow/#product-development-timeline",
    # "https://handbook.gitlab.com/handbook/enterprise-data/organization/programs/data-for-product-managers/",
    # "https://handbook.gitlab.com/handbook/company/pricing/",
    # "https://handbook.gitlab.com/handbook/acquisitions/",
    # "https://handbook.gitlab.com/handbook/product/ux/",
    "https://handbook.gitlab.com/handbook/legal/commercial/",
    "https://handbook.gitlab.com/handbook/legal/publiccompanyresources/",
    "https://handbook.gitlab.com/handbook/legal/employment-law/",
    "https://handbook.gitlab.com/handbook/legal/esg/",
    "https://handbook.gitlab.com/handbook/legal/legalops/",
    "https://handbook.gitlab.com/handbook/legal/privacy/",
    "https://handbook.gitlab.com/handbook/legal/product/",
    "https://handbook.gitlab.com/handbook/legal/risk-management-dispute-resolution/",
    "https://handbook.gitlab.com/handbook/legal/trade-compliance/"
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
            logger.error(f"Skipping URL due to error: {url} ({e})")

    output_file = os.path.join(out_dir, "handbook_paragraphs.json")
    save_json(items, output_file)
    logger.info(f" Scraper run completed successfully. Output file: {output_file}")
