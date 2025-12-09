#!/usr/bin/env python3
"""
Clean and preprocess YouTube transcripts
Handles common transcript artifacts and normalizes text
"""

import json
import re
from pathlib import Path
from datetime import datetime
import unicodedata

class TranscriptCleaner:
    def __init__(self):
        # Common patterns to clean
        self.patterns = {
            'timestamps': r'\[\d{2}:\d{2}\]|\(\d{2}:\d{2}\)',  # [00:00] or (00:00)
            'speaker_labels': r'\[.*?\]|\(.*?\)',  # [Speaker 1] or (Speaker 1)
            'multiple_spaces': r'\s+',  # Multiple spaces
            'special_chars': r'[♪|#|@|&|*]',  # Musical notes and special characters
            'html_tags': r'<[^>]+>',  # HTML tags
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'multiple_newlines': r'\n+',
        }
        
        # Common subtitle artifacts
        self.artifacts = [
            '[Music]', '[Applause]', '[Laughter]', 
            '(music)', '(applause)', '(laughter)',
            '♪', '►', '¶', '•'
        ]
    
    def normalize_unicode(self, text):
        """Normalize Unicode characters and remove control characters"""
        # Normalize to NFKC form (compatible decomposition)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters but keep newlines and tabs
        return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
    
    def fix_common_errors(self, text):
        """Fix common transcript errors and artifacts"""
        # Remove common artifacts
        for artifact in self.artifacts:
            text = text.replace(artifact, '')
        
        # Fix common OCR/transcript errors
        text = text.replace('|', 'I')  # Vertical bar often misrecognized as 'I'
        text = text.replace('0', 'O')  # Zero often misrecognized as 'O'
        text = text.replace('1', 'l')  # One often misrecognized as 'l'
        
        return text
    
    def clean_text(self, text):
        """Apply all cleaning steps to text"""
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Remove filler words
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        for word in filler_words:
            text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
        
        # Remove patterns using regex
        for pattern in self.patterns.values():
            text = re.sub(pattern, ' ', text)
        
        # Fix common errors
        text = self.fix_common_errors(text)
        
        # Final cleanup
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        
        return text
    
    def clean_segment(self, segment):
        """Clean an individual transcript segment"""
        cleaned_segment = segment.copy()  # Create a copy to preserve other fields
        cleaned_segment['text'] = self.clean_text(segment['text'])
        return cleaned_segment
    
    def merge_segments(self, segments):
        """Merge consecutive segments that make sense together"""
        if not segments:
            return []
            
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            # If segments are continuous in time and make sense to merge
            if next_seg['start'] == current['end']:
                current['text'] += ' ' + next_seg['text']
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged
    
    def process_transcript(self, transcript):
        """Process a transcript dictionary"""
        if 'segments' not in transcript:
            raise KeyError("Transcript must contain 'segments' key")
            
        cleaned_transcript = transcript.copy()
        
        # Clean segments if they exist
        cleaned_segments = self.merge_segments([
            self.clean_segment(segment)
            for segment in transcript['segments']
        ])
        
        cleaned_transcript['segments'] = cleaned_segments
        cleaned_transcript['segment_count'] = len(cleaned_segments)
        cleaned_transcript['word_count'] = sum(
            len(seg['text'].split()) 
            for seg in cleaned_segments
        )
        cleaned_transcript['cleaned_at'] = datetime.now().isoformat()
        
        # Create full text from merged segments
        full_text = ' '.join(seg['text'] for seg in cleaned_segments)
        cleaned_transcript['full_text'] = self.clean_text(full_text)
        
        return cleaned_transcript
    
    def process_file(self, input_path, output_path):
        """Process a transcript file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
            
        cleaned_transcript = self.process_transcript(transcript)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_transcript, f, indent=2)
            
        return cleaned_transcript

def process_directory(input_dir='data-pipeline/data/raw/transcripts', 
                     output_dir='data-pipeline/data/processed/cleaned'):
    """Process all transcripts in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cleaner = TranscriptCleaner()
    processed = 0
    total_files = len(list(input_path.glob('*.json')))
    
    print(f"Processing {total_files} transcripts...")
    print("="*60)
    
    for json_file in input_path.glob('*.json'):
        try:
            print(f"\nProcessing: {json_file.name}")
            
            # Read transcript
            with open(json_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Clean transcript
            cleaned_data = cleaner.clean_transcript(transcript_data)
            
            # Save cleaned transcript
            output_file = output_path / f"cleaned_{json_file.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            # Print stats
            orig_words = transcript_data['word_count']
            clean_words = cleaned_data['word_count']
            diff = orig_words - clean_words
            print(f"✓ Words: {orig_words:,} → {clean_words:,} ({diff:+,})")
            
            processed += 1
            
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {str(e)}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {total_files - processed}")
    print(f"\nOutput directory: {output_dir}")

def main():
    """Main function with example usage"""
    print("YouTube Transcript Cleaner")
    print("="*60)
    
    # Process all transcripts
    process_directory()
    
    print("\nNext steps:")
    print("1. Check data/processed/cleaned/ for cleaned transcripts")
    print("2. Each file contains:")
    print("   - Cleaned full text")
    print("   - Cleaned segments")
    print("   - Original metadata")
    print("   - Word counts before/after")

if __name__ == '__main__':
    main()