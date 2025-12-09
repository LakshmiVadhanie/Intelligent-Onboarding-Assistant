#!/usr/bin/env python3
"""
Test the GitLab Handbook Scraper
"""

import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path.cwd() / "data-pipeline"))

print("Testing GitLab Handbook Scraper...")
print("=" * 50)

# Test imports
try:
    from scripts.ingestion.gitlab_scraper import GitLabHandbookScraper
    print("✅ Scraper module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import scraper: {e}")
    print("\nMake sure you:")
    print("1. Created gitlab_scraper.py in data-pipeline/scripts/ingestion/")
    print("2. Installed required packages: pip install requests beautifulsoup4 tqdm")
    sys.exit(1)

# Test scraper initialization
try:
    scraper = GitLabHandbookScraper()
    print("✅ Scraper initialized successfully")
    print(f"   Base URL: {scraper.base_url}")
    print(f"   Output directory: {scraper.output_dir}")
except Exception as e:
    print(f"❌ Failed to initialize scraper: {e}")
    sys.exit(1)

# Ask user if they want to run a test scrape
print("\n" + "=" * 50)
print("Ready to test scraping!")
print("\n⚠️  This will scrape a few pages from GitLab Handbook")
print("It will save data to: data-pipeline/data/raw/handbook/")
response = input("\nDo you want to run a test scrape? (yes/no): ")

if response.lower() in ['yes', 'y']:
    print("\nStarting test scrape (5 pages max)...")
    try:
        # Run with very limited pages for testing
        documents = scraper.run(max_pages_per_section=2)
        print(f"\n✅ Test scrape successful!")
        print(f"📊 Scraped {len(documents)} documents")
        
        # Check if files were created
        handbook_dir = Path("data-pipeline/data/raw/handbook")
        json_files = list(handbook_dir.glob("*.json"))
        print(f"📁 Created {len(json_files)} files in {handbook_dir}")
        
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the GitLab handbook URL is accessible")
        print("3. Check the logs in data-pipeline/logs/pipeline/")
else:
    print("\nTest skipped. When you're ready, run:")
    print("python3 data-pipeline/scripts/ingestion/gitlab_scraper.py")