import re
from collections import Counter
from typing import Dict, List, Optional
import logging
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
