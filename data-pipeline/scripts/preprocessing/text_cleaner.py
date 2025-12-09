"""
Text Preprocessing Pipeline
Cleans and processes raw text data from various sources
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import unicodedata
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.config_loader import config
from scripts.utils.logging_config import pipeline_logger

class TextCleaner:
    """Cleans and normalizes text data"""
    
    def __init__(self):
        """Initialize text cleaner with configuration"""
        self.logger = pipeline_logger
        self.config = config
        
        # Load cleaning configuration
        self.remove_html = self.config.get("preprocessing.cleaning.remove_html", True)
        self.normalize_whitespace = self.config.get("preprocessing.cleaning.normalize_whitespace", True)
        self.min_text_length = self.config.get("preprocessing.cleaning.min_text_length", 50)
        
        # Set up paths
        self.input_dir = Path("data-pipeline/data/raw")
        self.output_dir = Path("data-pipeline/data/processed/cleaned")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Text Cleaner initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags if present
        if self.remove_html:
            text = self.remove_html_tags(text)
        
        # Normalize unicode characters
        text = self.normalize_unicode(text)
        
        # Fix common encoding issues
        text = self.fix_encoding_issues(text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.normalize_whitespace_func(text)
        
        # Remove excessive line breaks
        text = self.remove_excessive_linebreaks(text)
        
        # Remove special characters but keep punctuation
        text = self.clean_special_characters(text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Remove HTML entities
        clean_text = re.sub(r'&[a-zA-Z]+;', ' ', clean_text)
        clean_text = re.sub(r'&#[0-9]+;', ' ', clean_text)
        return clean_text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize to NFKD form
        text = unicodedata.normalize('NFKD', text)
        # Remove non-ASCII characters that are not important
        text = ''.join(char for char in text if ord(char) < 128 or char.isalpha())
        return text
    
    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '--',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            '\xa0': ' ',  # Non-breaking space
            '\u200b': '',  # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def normalize_whitespace_func(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Remove spaces before punctuation
        text = re.sub(r' +([.,!?;:])', r'\1', text)
        return text
    
    def remove_excessive_linebreaks(self, text: str) -> str:
        """Remove excessive line breaks"""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text
    
    def clean_special_characters(self, text: str) -> str:
        """Remove special characters but keep useful punctuation"""
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()\[\]{}/@#$%&*+=]', '', text)
        return text
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text meets minimum requirements"""
        if not text:
            return False
        if len(text) < self.min_text_length:
            return False
        # Check if text has actual content (not just special chars)
        if len(re.findall(r'\w+', text)) < 5:  # At least 5 words
            return False
        return True
    
    def process_handbook_document(self, doc_path: Path) -> Optional[Dict]:
        """
        Process a single handbook document
        
        Args:
            doc_path: Path to document JSON file
            
        Returns:
            Processed document or None if invalid
        """
        try:
            # Load document
            with open(doc_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Clean content
            original_content = document.get('content', '')
            cleaned_content = self.clean_text(original_content)
            
            # Validate cleaned content
            if not self.is_valid_text(cleaned_content):
                self.logger.warning(f"Document {doc_path.name} has insufficient content after cleaning")
                return None
            
            # Update document with cleaned content
            processed_doc = {
                'content': cleaned_content,
                'metadata': document.get('metadata', {}),
                'processing': {
                    'original_length': len(original_content),
                    'cleaned_length': len(cleaned_content),
                    'cleaned_at': datetime.now().isoformat(),
                    'processor': 'text_cleaner_v1'
                }
            }
            
            # Add word count
            word_count = len(cleaned_content.split())
            processed_doc['processing']['word_count'] = word_count
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Failed to process {doc_path}: {e}")
            return None
    
    def process_all_handbook_documents(self):
        """Process all handbook documents"""
        self.logger.info("Starting handbook text cleaning")
        
        # Find all handbook JSON files
        handbook_dir = self.input_dir / "handbook"
        json_files = list(handbook_dir.glob("handbook_*.json"))
        
        if not json_files:
            self.logger.warning("No handbook files found to process")
            return []
        
        self.logger.info(f"Found {len(json_files)} handbook documents to process")
        
        processed_docs = []
        failed_docs = []
        
        # Process each document
        for doc_path in tqdm(json_files, desc="Cleaning documents"):
            processed_doc = self.process_handbook_document(doc_path)
            
            if processed_doc:
                # Save processed document
                output_path = self.output_dir / f"cleaned_{doc_path.name}"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_doc, f, ensure_ascii=False, indent=2)
                
                processed_docs.append(processed_doc)
                self.logger.info(f"Saved cleaned document to {output_path}")
            else:
                failed_docs.append(doc_path.name)
        
        # Save processing summary
        summary = {
            'total_documents': len(json_files),
            'successfully_processed': len(processed_docs),
            'failed': len(failed_docs),
            'failed_documents': failed_docs,
            'processed_at': datetime.now().isoformat(),
            'average_word_count': sum(d['processing']['word_count'] for d in processed_docs) / len(processed_docs) if processed_docs else 0
        }
        
        summary_path = self.output_dir / "cleaning_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Text cleaning complete! Processed: {len(processed_docs)}, Failed: {len(failed_docs)}")
        self.logger.info(f"Summary saved to {summary_path}")
        
        return processed_docs

def main():
    """Main function to run text cleaning"""
    cleaner = TextCleaner()
    processed_docs = cleaner.process_all_handbook_documents()
    
    print(f"\n✅ Text cleaning complete!")
    print(f"📁 Cleaned data saved to: data-pipeline/data/processed/cleaned/")
    print(f"📊 Total documents processed: {len(processed_docs)}")

if __name__ == "__main__":
    main()