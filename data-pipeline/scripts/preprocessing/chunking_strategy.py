"""
Text Chunking Strategy
Splits cleaned documents into smaller chunks for embedding
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.config_loader import config
from scripts.utils.logging_config import pipeline_logger

class TextChunker:
    """Chunks text documents into smaller pieces"""
    
    def __init__(self):
        """Initialize text chunker with configuration"""
        self.logger = pipeline_logger
        self.config = config
        
        # Load chunking configuration
        self.method = self.config.get("preprocessing.chunking.method", "recursive")
        self.chunk_size = self.config.get("preprocessing.chunking.chunk_size", 800)
        self.chunk_overlap = self.config.get("preprocessing.chunking.chunk_overlap", 150)
        self.min_chunk_size = self.config.get("preprocessing.chunking.min_chunk_size", 100)
        
        # Set up paths
        self.input_dir = Path("data-pipeline/data/processed/cleaned")
        self.output_dir = Path("data-pipeline/data/processed/chunked")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Text Chunker initialized - Method: {self.method}, Size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text
        Rough estimate: 1 token ≈ 4 characters
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def recursive_chunk(self, text: str, separators: List[str] = None) -> List[str]:
        """
        Recursively split text into chunks
        
        Args:
            text: Text to chunk
            separators: List of separators to use
            
        Returns:
            List of text chunks
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]
        
        chunks = []
        current_chunk = ""
        
        # Try to split by the first separator
        if not separators:
            # No more separators, split by character count
            words = text.split()
            for word in words:
                test_chunk = current_chunk + " " + word if current_chunk else word
                if self.estimate_tokens(test_chunk) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk = test_chunk
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        separator = separators[0]
        splits = text.split(separator)
        
        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            if i < len(splits) - 1:
                split = split + separator
            
            test_chunk = current_chunk + split
            
            if self.estimate_tokens(test_chunk) > self.chunk_size:
                # Current chunk is too big
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Check if the split itself is too big
                if self.estimate_tokens(split) > self.chunk_size:
                    # Recursively chunk with next separator
                    sub_chunks = self.recursive_chunk(split, separators[1:])
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
            else:
                current_chunk = test_chunk
        
        # Add remaining chunk
        if current_chunk and self.estimate_tokens(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks for context continuity
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - add beginning of next chunk
                if i + 1 < len(chunks):
                    next_chunk_words = chunks[i + 1].split()
                    overlap_words = next_chunk_words[:self.chunk_overlap // 10]  # Rough word count
                    chunk_with_overlap = chunk + " " + " ".join(overlap_words)
                else:
                    chunk_with_overlap = chunk
            elif i == len(chunks) - 1:
                # Last chunk - add end of previous chunk
                prev_chunk_words = chunks[i - 1].split()
                overlap_words = prev_chunk_words[-self.chunk_overlap // 10:]
                chunk_with_overlap = " ".join(overlap_words) + " " + chunk
            else:
                # Middle chunks - add both overlaps
                prev_chunk_words = chunks[i - 1].split()
                next_chunk_words = chunks[i + 1].split()
                prev_overlap = prev_chunk_words[-self.chunk_overlap // 20:]
                next_overlap = next_chunk_words[:self.chunk_overlap // 20:]
                chunk_with_overlap = " ".join(prev_overlap) + " " + chunk + " " + " ".join(next_overlap)
            
            overlapped_chunks.append(chunk_with_overlap.strip())
        
        return overlapped_chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a single document
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            List of chunk dictionaries
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        if not content:
            return []
        
        # Perform chunking
        if self.method == "recursive":
            chunks = self.recursive_chunk(content)
        else:
            # Simple fixed-size chunking as fallback
            chunks = self.simple_chunk(content)
        
        # Add overlap if configured
        if self.chunk_overlap > 0:
            chunks = self.add_overlap(chunks)
        
        # Create chunk documents
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                'chunk_id': f"{metadata.get('url', 'unknown')}_{i}",
                'content': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_method': self.method,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                },
                'processing': {
                    'chunked_at': datetime.now().isoformat(),
                    'estimated_tokens': self.estimate_tokens(chunk_text),
                    'character_count': len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def simple_chunk(self, text: str) -> List[str]:
        """
        Simple fixed-size chunking
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) // 4  # Rough token estimate
            if current_size + word_size > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_all_documents(self):
        """Process all cleaned documents"""
        self.logger.info("Starting document chunking")
        
        # Find all cleaned JSON files
        json_files = list(self.input_dir.glob("cleaned_*.json"))
        
        if not json_files:
            self.logger.warning("No cleaned files found to process")
            return []
        
        self.logger.info(f"Found {len(json_files)} cleaned documents to chunk")
        
        all_chunks = []
        chunk_stats = []
        
        # Process each document
        for doc_path in tqdm(json_files, desc="Chunking documents"):
            try:
                # Load document
                with open(doc_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                # Chunk document
                chunks = self.chunk_document(document)
                
                if chunks:
                    # Save chunks
                    output_path = self.output_dir / f"chunks_{doc_path.stem}.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    
                    all_chunks.extend(chunks)
                    chunk_stats.append({
                        'document': doc_path.name,
                        'num_chunks': len(chunks),
                        'avg_chunk_size': sum(c['processing']['word_count'] for c in chunks) / len(chunks)
                    })
                    
                    self.logger.info(f"Created {len(chunks)} chunks from {doc_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to chunk {doc_path}: {e}")
        
        # Save chunking summary
        summary = {
            'total_documents': len(json_files),
            'total_chunks': len(all_chunks),
            'average_chunks_per_doc': len(all_chunks) / len(json_files) if json_files else 0,
            'chunk_statistics': chunk_stats,
            'chunked_at': datetime.now().isoformat(),
            'chunking_config': {
                'method': self.method,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }
        
        summary_path = self.output_dir / "chunking_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Chunking complete! Total chunks: {len(all_chunks)}")
        self.logger.info(f"Summary saved to {summary_path}")
        
        return all_chunks

def main():
    """Main function to run chunking"""
    chunker = TextChunker()
    chunks = chunker.process_all_documents()
    
    print(f"\n✅ Document chunking complete!")
    print(f"📁 Chunks saved to: data-pipeline/data/processed/chunked/")
    print(f"📊 Total chunks created: {len(chunks)}")

if __name__ == "__main__":
    main()