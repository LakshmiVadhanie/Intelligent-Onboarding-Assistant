"""
Blog Fetcher for GitLab Blog Posts
Fetches blog posts from GitLab's blog for onboarding content
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

import requests
import feedparser
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.config_loader import config
from scripts.utils.logging_config import pipeline_logger

class BlogFetcher:
    """Fetches blog posts from GitLab's blog"""
    
    def __init__(self):
        """Initialize the blog fetcher"""
        self.logger = pipeline_logger
        self.config = config
        
        # Load configuration
        self.rss_url = self.config.get("data_sources.blog.rss_url", "https://about.gitlab.com/atom.xml")
        self.base_url = self.config.get("data_sources.blog.base_url", "https://about.gitlab.com/blog/")
        self.max_posts = self.config.get("data_sources.blog.max_posts", 20)
        
        # Set up storage paths
        self.output_dir = Path("data-pipeline/data/raw/blogs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed posts
        self.processed_posts = []
        self.failed_posts = []
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Onboarding Assistant Bot)'
        })
        
        self.logger.info(f"Blog Fetcher initialized. RSS URL: {self.rss_url}")
    
    def fetch_rss_feed(self) -> Optional[Dict]:
        """
        Fetch and parse RSS feed
        
        Returns:
            Parsed feed data or None
        """
        try:
            self.logger.info(f"Fetching RSS feed from {self.rss_url}")
            
            # Fetch RSS feed
            response = self.session.get(self.rss_url, timeout=30)
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.text)
            
            if feed.bozo:
                self.logger.warning(f"RSS feed parsing had issues: {feed.bozo_exception}")
            
            return feed
            
        except Exception as e:
            self.logger.error(f"Failed to fetch RSS feed: {e}")
            return None
    
    def extract_content_from_url(self, url: str) -> Optional[str]:
        """
        Extract main content from blog post URL
        
        Args:
            url: Blog post URL
            
        Returns:
            Extracted content or None
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Try to find main content area (GitLab specific selectors)
            content_selectors = [
                'article',
                '.blog-post-content',
                '.post-content',
                '.content',
                'main',
                '[role="main"]'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body
            
            if main_content:
                # Get text and clean it
                text = main_content.get_text(separator='\n', strip=True)
                # Clean up excessive whitespace
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                return '\n'.join(lines)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    def is_relevant_post(self, entry: Dict) -> bool:
        """
        Check if blog post is relevant for onboarding
        
        Args:
            entry: RSS feed entry
            
        Returns:
            True if relevant, False otherwise
        """
        # Keywords that indicate onboarding-relevant content
        relevant_keywords = [
            'onboarding', 'culture', 'remote', 'handbook', 'values',
            'team', 'workflow', 'process', 'engineering', 'devops',
            'collaboration', 'communication', 'transparency', 'iteration',
            'efficiency', 'results', 'diversity', 'inclusion', 'belong',
            'career', 'growth', 'learning', 'development', 'hiring'
        ]
        
        # Check title and summary
        title = entry.get('title', '').lower()
        summary = entry.get('summary', '').lower()
        
        # Check if any relevant keyword is present
        for keyword in relevant_keywords:
            if keyword in title or keyword in summary:
                return True
        
        # Check categories/tags
        tags = entry.get('tags', [])
        for tag in tags:
            tag_term = tag.get('term', '').lower()
            for keyword in relevant_keywords:
                if keyword in tag_term:
                    return True
        
        return False
    
    def process_blog_post(self, entry: Dict) -> Optional[Dict]:
        """
        Process a single blog post entry
        
        Args:
            entry: RSS feed entry
            
        Returns:
            Processed blog post data or None
        """
        try:
            title = entry.get('title', 'Untitled')
            link = entry.get('link', '')
            
            if not link:
                self.logger.warning(f"No link for post: {title}")
                return None
            
            self.logger.info(f"Processing blog post: {title}")
            
            # Extract content from the blog post page
            content = self.extract_content_from_url(link)
            
            if not content or len(content) < 100:  # Skip if too short
                self.logger.warning(f"Insufficient content for: {title}")
                return None
            
            # Extract metadata
            metadata = {
                'title': title,
                'url': link,
                'published': entry.get('published', ''),
                'updated': entry.get('updated', ''),
                'author': entry.get('author', ''),
                'categories': [tag.get('term', '') for tag in entry.get('tags', [])],
                'summary': entry.get('summary', '')[:500],  # First 500 chars
                'fetched_at': datetime.now().isoformat(),
                'source': 'gitlab_blog'
            }
            
            # Create blog post document
            blog_data = {
                'content': content,
                'metadata': metadata,
                'processing': {
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'is_relevant': self.is_relevant_post(entry)
                }
            }
            
            return blog_data
            
        except Exception as e:
            self.logger.error(f"Failed to process blog post: {e}")
            return None
    
    def save_blog_post(self, blog_data: Dict, filename: str):
        """
        Save blog post data to disk
        
        Args:
            blog_data: Blog post data
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(blog_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved blog post to {output_path}")
    
    def run(self):
        """Run the blog fetching process"""
        self.logger.info("Starting blog fetching process")
        start_time = time.time()
        
        # Fetch RSS feed
        feed = self.fetch_rss_feed()
        
        if not feed or not feed.entries:
            self.logger.error("No feed entries found")
            return []
        
        self.logger.info(f"Found {len(feed.entries)} blog posts in feed")
        
        # Process entries (limit to max_posts)
        entries_to_process = feed.entries[:self.max_posts]
        
        # Filter for relevant posts if we have more than needed
        if len(feed.entries) > self.max_posts:
            # Prioritize relevant posts
            relevant_entries = [e for e in feed.entries if self.is_relevant_post(e)]
            other_entries = [e for e in feed.entries if not self.is_relevant_post(e)]
            
            entries_to_process = relevant_entries[:self.max_posts]
            if len(entries_to_process) < self.max_posts:
                remaining = self.max_posts - len(entries_to_process)
                entries_to_process.extend(other_entries[:remaining])
        
        self.logger.info(f"Processing {len(entries_to_process)} blog posts")
        
        # Process each blog post
        for entry in tqdm(entries_to_process, desc="Fetching blog posts"):
            blog_data = self.process_blog_post(entry)
            
            if blog_data:
                # Generate filename from URL
                url_hash = hashlib.md5(blog_data['metadata']['url'].encode()).hexdigest()[:8]
                filename = f"blog_{url_hash}.json"
                
                # Save blog post
                self.save_blog_post(blog_data, filename)
                self.processed_posts.append({
                    'title': blog_data['metadata']['title'],
                    'url': blog_data['metadata']['url'],
                    'relevant': blog_data['processing']['is_relevant']
                })
                
                # Rate limiting
                time.sleep(0.5)  # Be respectful to the server
            else:
                self.failed_posts.append({
                    'title': entry.get('title', 'Unknown'),
                    'url': entry.get('link', 'Unknown')
                })
        
        # Save summary
        summary = {
            'total_posts': len(entries_to_process),
            'successfully_processed': len(self.processed_posts),
            'failed': len(self.failed_posts),
            'relevant_posts': sum(1 for p in self.processed_posts if p.get('relevant', False)),
            'processed_posts': self.processed_posts,
            'failed_posts': self.failed_posts,
            'processed_at': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time
        }
        
        summary_path = self.output_dir / "blog_fetching_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Blog fetching complete! Processed: {len(self.processed_posts)}, Failed: {len(self.failed_posts)}")
        self.logger.info(f"Summary saved to {summary_path}")
        
        return self.processed_posts

def main():
    """Main function to run the blog fetcher"""
    fetcher = BlogFetcher()
    processed = fetcher.run()
    
    print(f"\n✅ Blog fetching complete!")
    print(f"📁 Blog posts saved to: data-pipeline/data/raw/blogs/")
    print(f"📊 Posts processed: {len(processed)}")
    
    # Show relevant posts
    relevant = [p for p in processed if p.get('relevant', False)]
    if relevant:
        print(f"📌 Relevant posts for onboarding: {len(relevant)}")

if __name__ == "__main__":
    main()