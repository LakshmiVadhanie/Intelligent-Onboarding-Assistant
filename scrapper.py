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


# ---------------------- Utility Functions ---------------------- #

def fetch_page_content(url: str) -> str:
    """
    Fetch raw HTML content from a given URL.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: Raw HTML text of the page.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GitLabScraper/1.0; +https://handbook.gitlab.com)"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def parse_html_content(html: str) -> Dict[str, Any]:
    """
    Parse the HTML and extract structured content from the GitLab handbook page.

    Args:
        html (str): The raw HTML string.

    Returns:
        dict: A structured dictionary containing:
            - "title" (str): The main page title.
            - "headings" (List[dict]): List of headings (H1–H3) with their text.
            - "paragraphs" (List[str]): Clean text paragraphs.
            - "links" (List[str]): Extracted URLs from within the content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Page title
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"

    # Extract headings
    headings = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            headings.append({"level": level, "text": tag.get_text(strip=True)})

    # Extract paragraphs
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]

    # Extract internal and external links
    links = [a["href"] for a in soup.find_all("a", href=True)]

    return {
        "title": title,
        "headings": headings,
        "paragraphs": paragraphs,
        "links": links
    }


def scrape_gitlab_handbook(url: str) -> Dict[str, Any]:
    """
    Orchestrates the full scraping pipeline: fetch → parse → return structured data.

    Args:
        url (str): GitLab handbook URL to scrape.

    Returns:
        dict: Structured dictionary with the following keys:
            {
                "url": str,              # Original URL
                "title": str,            # Main page title
                "headings": [            # List of heading tags
                    {"level": "h2", "text": "Iteration"},
                    {"level": "h3", "text": "Small changes > big releases"}
                ],
                "paragraphs": [          # Clean list of paragraph text
                    "At GitLab, we value iteration because...",
                    "Deliver small, valuable changes quickly..."
                ],
                "links": [               # All internal & external URLs
                    "https://about.gitlab.com",
                    "/handbook/engineering/",
                    ...
                ]
            }

    Raises:
        Exception: Propagates exceptions from fetch_page_content() or parse_html_content().
    """
    html = fetch_page_content(url)
    parsed_data = parse_html_content(html)
    parsed_data["url"] = url
    return parsed_data


# ---------------------- Example Run ---------------------- #
if __name__ == "__main__":
    test_url = "https://handbook.gitlab.com/handbook/values/"
    result = scrape_gitlab_handbook(test_url)

    print("\n=== SCRAPED PAGE SUMMARY ===")
    print(f"Title: {result['title']}\n")
    print(f"Total Headings: {len(result['headings'])}")
    print(f"Total Paragraphs: {len(result['paragraphs'])}")
    print(f"Total Links: {len(result['links'])}\n")

    # Preview first few paragraphs
    for para in result["paragraphs"][:5]:
        print("-", para[:120], "...")
