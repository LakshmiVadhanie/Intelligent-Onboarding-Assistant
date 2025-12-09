import re
import sys
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional
import logging

# Add the data-pipeline directory to the Python path
current_dir = Path(__file__).resolve().parent
data_pipeline_dir = current_dir.parent.parent
sys.path.insert(0, str(data_pipeline_dir))

from scripts.utils.logging_setup import setup_logger

logger = setup_logger('bias_detector')

class BiasDetector:
    def __init__(self, sensitive_terms: Optional[List[str]] = None):
        # List of sensitive or potentially biased terms to check for
        self.sensitive_terms = sensitive_terms or [
            'he', 'she', 'him', 'her', 'man', 'woman', 'guys', 'girls',
            'black', 'white', 'asian', 'indian', 'latino', 'gay', 'straight',
            'disabled', 'handicapped', 'crazy', 'insane', 'old', 'young'
        ]

    def term_frequency(self, text: str) -> Dict[str, int]:
        """Count frequency of sensitive terms in text"""
        text = text.lower()
        freq = {term: len(re.findall(r'\b' + re.escape(term) + r'\b', text)) for term in self.sensitive_terms}
        return freq

    def detect_bias(self, transcript: Dict) -> Dict[str, int]:
        """Detect bias in a transcript dict (expects 'segments' or 'full_text')"""
        if 'full_text' in transcript:
            text = transcript['full_text']
        elif 'segments' in transcript:
            text = ' '.join(seg.get('text', '') for seg in transcript['segments'])
        else:
            logger.warning('Transcript missing text fields for bias detection.')
            return {}
        freq = self.term_frequency(text)
        # Log any nonzero frequencies
        for term, count in freq.items():
            if count > 0:
                logger.warning(f"Sensitive term '{term}' found {count} times.")
        return freq

    def speaker_balance(self, transcript: Dict) -> Dict[str, int]:
        """Check for speaker label balance if available (expects 'speaker' in segments)"""
        if 'segments' not in transcript:
            return {}
        speakers = [seg.get('speaker') for seg in transcript['segments'] if 'speaker' in seg]
        speaker_counts = dict(Counter(speakers))
        if speaker_counts:
            logger.info(f"Speaker distribution: {speaker_counts}")
        return speaker_counts


if __name__ == "__main__":
    # Test with sample transcript
    sample_transcript = {
        'full_text': 'Hello guys, this is a test. The man and woman were discussing the project. She said he was right.',
        'segments': [
            {'text': 'Hello guys, this is a test.', 'speaker': 'Speaker_1'},
            {'text': 'The man and woman were discussing the project.', 'speaker': 'Speaker_2'},
            {'text': 'She said he was right.', 'speaker': 'Speaker_1'}
        ]
    }
    
    print("=" * 60)
    print("BIAS DETECTOR TEST")
    print("=" * 60)
    
    detector = BiasDetector()
    
    # Detect bias
    bias_freq = detector.detect_bias(sample_transcript)
    print("\nBias Detection Results:")
    print("-" * 40)
    for term, count in bias_freq.items():
        if count > 0:
            print(f"  '{term}': {count} occurrences")
    
    # Check speaker balance
    speaker_balance = detector.speaker_balance(sample_transcript)
    print("\nSpeaker Balance:")
    print("-" * 40)
    for speaker, count in speaker_balance.items():
        print(f"  {speaker}: {count} segments")
    
    print("=" * 60)
