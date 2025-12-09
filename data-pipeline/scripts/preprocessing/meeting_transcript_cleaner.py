#!/usr/bin/env python3
"""
Specialized cleaner for meeting transcripts
Handles common meeting transcript artifacts and makes text more readable
"""

import json
import re
from pathlib import Path
from datetime import datetime
import unicodedata

class MeetingTranscriptCleaner:
    def __init__(self):
        # Common patterns to clean
        self.patterns = {
            'timestamps': r'\[\d{2}:\d{2}\]|\(\d{2}:\d{2}\)',  # [00:00] or (00:00)
            'speaker_markers': r'^.*?:\s',  # "Speaker Name: " at start of line
            'multiple_spaces': r'\s+',  # Multiple spaces
            'special_chars': r'[♪|#|@|&|*]',  # Musical notes and special characters
            'html_tags': r'<[^>]+>',  # HTML tags
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'multiple_newlines': r'\n+',
        }
        
        # Common filler words and phrases to remove
        self.filler_words = [
            r'\bum\b', r'\buh\b', r'\blike\b', r'\byou know\b', r'\bkind of\b', r'\bsort of\b',
            r'\bI mean\b', r'\bright\b', r'\bso\b', r'\bwell\b', r'\bokay\b', r'\bbasically\b',
            r'\bactually\b', r'\bliterally\b', r'\banyway[s]?\b', r'\bI guess\b'
        ]
        
        # Meeting-specific artifacts
        self.artifacts = [
            '[Music]', '[Applause]', '[Laughter]', '[Background Noise]',
            '(music)', '(applause)', '(laughter)', '(noise)',
            '♪', '►', '¶', '•'
        ]
        
        # Transition phrases to clean up
        self.transitions = [
            "let me", "I'm going to", "we're going to", "going to",
            "let's see", "you see", "you know what"
        ]

    def clean_text(self, text):
        """Apply all cleaning steps to text"""
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove common patterns
        for pattern in self.patterns.values():
            text = re.sub(pattern, ' ', text)
        
        # Remove artifacts
        for artifact in self.artifacts:
            text = text.replace(artifact, '')
        
        # Remove filler words
        for filler in self.filler_words:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        
        # Clean transition phrases
        for transition in self.transitions:
            text = re.sub(r'\b' + transition + r'\b', '', text, flags=re.IGNORECASE)
        
        # Fix common transcription errors
        text = text.replace('|', 'I')  # Vertical bar often misrecognized as 'I'
        text = text.replace('0', 'O')  # Zero often misrecognized as 'O'
        text = text.replace('1', 'l')  # One often misrecognized as 'l'
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def clean_segment(self, segment):
        """Clean an individual transcript segment"""
        cleaned_segment = segment.copy()
        cleaned_segment['text'] = self.clean_text(segment['text'])
        return cleaned_segment
    
    def merge_consecutive_segments(self, segments, max_gap=2.0):
        """Merge consecutive segments from same speaker if they're close in time"""
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            # If segments are close in time, merge them
            if next_seg['start'] - current['end'] <= max_gap:
                current['end'] = next_seg['end']
                current['text'] += ' ' + next_seg['text']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged

    def clean_transcript(self, transcript_data):
        """Clean and process meeting transcript"""
        cleaned_data = transcript_data.copy()
        
        # Clean and merge segments
        cleaned_segments = [self.clean_segment(seg) for seg in transcript_data['segments']]
        cleaned_segments = self.merge_consecutive_segments(cleaned_segments)
        
        # Update full text and segments
        cleaned_data['segments'] = cleaned_segments
        cleaned_data['full_text'] = ' '.join(seg['text'] for seg in cleaned_segments)
        
        # Update metadata
        cleaned_data['word_count'] = len(cleaned_data['full_text'].split())
        cleaned_data['segment_count'] = len(cleaned_segments)
        cleaned_data['cleaned_at'] = datetime.now().isoformat()
        
        return cleaned_data

def process_directory(input_dir='data-pipeline/data/raw/transcripts', 
                     output_dir='data-pipeline/data/processed/cleaned'):
    """Process all transcripts in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cleaner = MeetingTranscriptCleaner()
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
            orig_words = len(transcript_data['transcript'].split())
            clean_words = cleaned_data['word_count']
            diff = orig_words - clean_words
            orig_segs = len(transcript_data['segments'])
            clean_segs = len(cleaned_data['segments'])
            print(f"✓ Words: {orig_words:,} → {clean_words:,} ({diff:+,})")
            print(f"✓ Segments: {orig_segs:,} → {clean_segs:,}")
            
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
    print("Meeting Transcript Cleaner")
    print("="*60)
    
    # Process all transcripts
    process_directory()
    
    print("\nNext steps:")
    print("1. Check data-pipeline/data/processed/cleaned/ for cleaned transcripts")
    print("2. Each file contains:")
    print("   - Cleaned full text")
    print("   - Cleaned and merged segments")
    print("   - Original metadata")
    print("   - Word counts before/after")

if __name__ == '__main__':
    main()