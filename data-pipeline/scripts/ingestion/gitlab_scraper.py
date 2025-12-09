"""
GitLab Handbook Scraper
Scrapes and downloads GitLab handbook pages for the onboarding assistant
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.config_loader import config
from scripts.utils.logging_config import pipeline_logger

class GitLabHandbookScraper:
    """Scrapes GitLab handbook documentation"""
    
    def __init__(self):
        """Initialize the scraper with configuration"""
        self.logger = pipeline_logger
        self.config = config
        
        # Load configuration
        self.base_url = self.config.get("data_sources.gitlab.base_url")
        self.sections = self.config.get("data_sources.gitlab.sections_to_scrape", [])
        self.rate_limit = self.config.get("data_sources.gitlab.rate_limit", 2)
        self.timeout = self.config.get("data_sources.gitlab.timeout", 30)
        self.retry_attempts = self.config.get("data_sources.gitlab.retry_attempts", 3)
        
        # Set up storage paths
        self.output_dir = Path("data-pipeline/data/raw/handbook")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track scraped URLs to avoid duplicates
        self.scraped_urls = set()
        self.failed_urls = []
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Onboarding Assistant Bot)'
        })
        
        self.logger.info(f"GitLab Handbook Scraper initialized. Base URL: {self.base_url}")
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        """
        Scrape a single page
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with page content and metadata
        """
        try:
            # Check if already scraped
            if url in self.scraped_urls:
                return None
            
            self.logger.info(f"Scraping: {url}")
            
            # Retry logic
            for attempt in range(self.retry_attempts):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    self.logger.warning(f"Retry {attempt + 1}/{self.retry_attempts} for {url}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            content = self.extract_content(soup)
            
            # Extract metadata
            metadata = {
                "url": url,
                "title": self.extract_title(soup),
                "scraped_at": datetime.now().isoformat(),
                "content_length": len(content),
                "section": self.get_section_from_url(url)
            }
            
            # Create document
            document = {
                "content": content,
                "metadata": metadata,
                "html": response.text  # Keep original HTML for later processing
            }
            
            # Mark as scraped
            self.scraped_urls.add(url)
            
            # Rate limiting
            time.sleep(1.0 / self.rate_limit)
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            self.failed_urls.append({"url": url, "error": str(e)})
            return None
    
    def extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from the page
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Try to find main content area
        content_selectors = [
            'main',
            'article',
            '.content',
            '#content',
            '.handbook-content',
            '.documentation'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.body
        
        if main_content:
            # Get text and clean it
            text = main_content.get_text(separator='\n', strip=True)
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)
        
        return ""
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        # Try different title selectors
        title = soup.find('title')
        if title:
            return title.text.strip()
        
        h1 = soup.find('h1')
        if h1:
            return h1.text.strip()
        
        return "Untitled"
    
    def get_section_from_url(self, url: str) -> str:
        """Determine section from URL"""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if path_parts:
            return path_parts[0]
        return "general"
    
    def discover_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """
        Discover other handbook links from the page
        
        Args:
            url: Current page URL
            soup: BeautifulSoup object
            
        Returns:
            List of discovered URLs
        """
        discovered = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Make absolute URL
            absolute_url = urljoin(url, href)
            
            # Check if it's a handbook URL
            if self.base_url in absolute_url and absolute_url not in self.scraped_urls:
                # Filter out non-content pages
                skip_patterns = ['#', '.pdf', '.zip', 'mailto:', 'javascript:']
                if not any(pattern in absolute_url.lower() for pattern in skip_patterns):
                    discovered.append(absolute_url)
        
        return list(set(discovered))  # Remove duplicates
    
    def save_document(self, document: Dict, filename: str):
        """
        Save document to disk
        
        Args:
            document: Document dictionary
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved document to {output_path}")
    
    def scrape_section(self, section_url: str, max_pages: int = 50) -> List[Dict]:
        """
        Scrape a section of the handbook
        
        Args:
            section_url: Starting URL for the section
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of scraped documents
        """
        documents = []
        urls_to_scrape = [section_url]
        
        with tqdm(total=max_pages, desc="Scraping pages") as pbar:
            while urls_to_scrape and len(documents) < max_pages:
                url = urls_to_scrape.pop(0)
                
                # Skip if already scraped
                if url in self.scraped_urls:
                    continue
                
                # Scrape the page
                document = self.scrape_page(url)
                
                if document:
                    documents.append(document)
                    
                    # Save immediately
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"handbook_{url_hash}.json"
                    self.save_document(document, filename)
                    
                    # Discover new links
                    soup = BeautifulSoup(document['html'], 'html.parser')
                    new_links = self.discover_links(url, soup)
                    
                    # Add new links to queue (limit to same section)
                    for link in new_links:
                        if self.get_section_from_url(link) == self.get_section_from_url(url):
                            if link not in urls_to_scrape and link not in self.scraped_urls:
                                urls_to_scrape.append(link)
                    
                    pbar.update(1)
        
        return documents
    
    def run(self, max_pages_per_section: int = 20):
        """
        Run the complete scraping process
        
        Args:
            max_pages_per_section: Maximum pages to scrape per section
        """
        self.logger.info("Starting GitLab Handbook scraping")
        start_time = time.time()
        
        all_documents = []
        
        # Scrape each configured section
        for section_path in self.sections:
            section_url = urljoin(self.base_url, section_path)
            self.logger.info(f"Scraping section: {section_url}")
            
            documents = self.scrape_section(section_url, max_pages_per_section)
            all_documents.extend(documents)
            
            self.logger.info(f"Scraped {len(documents)} pages from {section_path}")
        
        # Save summary
        summary = {
            "total_pages": len(all_documents),
            "failed_pages": len(self.failed_urls),
            "sections_scraped": self.sections,
            "scraped_at": datetime.now().isoformat(),
            "duration_seconds": time.time() - start_time,
            "failed_urls": self.failed_urls
        }
        
        summary_path = self.output_dir / "scraping_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Scraping complete! Total pages: {len(all_documents)}, Failed: {len(self.failed_urls)}")
        self.logger.info(f"Summary saved to {summary_path}")
        
        return all_documents

def main():
    """Main function to run the scraper"""
    scraper = GitLabHandbookScraper()
    
    # Run with limited pages for testing
    # Increase max_pages_per_section for production
    documents = scraper.run(max_pages_per_section=5)  # Start small for testing
    
    print(f"\n✅ Scraping complete!")
    print(f"📁 Data saved to: data-pipeline/data/raw/handbook/")
    print(f"📊 Total documents: {len(documents)}")

if __name__ == "__main__":
    main()