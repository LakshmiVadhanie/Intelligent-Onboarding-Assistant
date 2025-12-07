"""
Text Preprocessing Utilities
============================

Used by scraper.py, transcription.py, etc.
Performs HTML decoding, whitespace cleanup, emoji removal, deduplication, and length filtering.
"""

import re
import html
import unicodedata
from typing import List, Dict, Any
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")


class ParagraphPreprocessor:
    def __init__(
        self,
        min_length: int = 30,
        remove_emojis: bool = True,
        dedupe: bool = True,
        collapse_whitespace: bool = True,
        lowercase: bool = False,
    ) -> None:
        self.min_length = min_length
        self.remove_emojis = remove_emojis
        self.dedupe = dedupe
        self.collapse_whitespace = collapse_whitespace
        self.lowercase = lowercase

        self._ws_re = re.compile(r"\s+")
        self._emoji_re = re.compile(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]")

        logger.info(
            f"ParagraphPreprocessor initialized "
            f"(min_length={min_length}, remove_emojis={remove_emojis}, dedupe={dedupe}, "
            f"collapse_whitespace={collapse_whitespace}, lowercase={lowercase})"
        )

    def _clean_text(self, text: str) -> str:
        """Internal cleaning pipeline."""
        s = html.unescape(text)
        s = unicodedata.normalize("NFKC", s)
        if self.lowercase:
            s = s.lower()
        if self.remove_emojis:
            s = self._emoji_re.sub("", s)
        if self.collapse_whitespace:
            s = self._ws_re.sub(" ", s)
        return s.strip()

    def preprocess_title(self, title: str) -> str:
        if not title:
            logger.warning("Empty title encountered during preprocessing.")
            return ""
        cleaned_title = self._clean_text(title)
        logger.debug(f"Preprocessed title: {cleaned_title[:60]}...")
        return cleaned_title

    def preprocess_paragraphs(self, paragraphs: List[str]) -> List[str]:
        if not paragraphs:
            logger.warning("No paragraphs provided for preprocessing.")
            return []

        logger.info(f"Starting preprocessing of {len(paragraphs)} paragraphs...")
        seen = set()
        cleaned: List[str] = []
        removed_short = removed_dupes = 0

        for p in paragraphs:
            if not p:
                continue
            cp = self._clean_text(p)
            if not cp or len(cp) < self.min_length:
                removed_short += 1
                continue
            if self.dedupe and cp in seen:
                removed_dupes += 1
                continue
            seen.add(cp)
            cleaned.append(cp)

        logger.info(
            f"Preprocessing complete: {len(cleaned)} kept, "
            f"{removed_short} short, {removed_dupes} duplicates removed."
        )
        return cleaned

    def preprocess_for_scraper(self, title: str, paragraphs: List[str]) -> Dict[str, Any]:
        """Convenience wrapper for scraper.py"""
        logger.info("Running preprocess_for_scraper()...")
        result = {
            "title": self.preprocess_title(title),
            "paragraphs": self.preprocess_paragraphs(paragraphs),
        }
        logger.info(f"Final preprocessed paragraph count: {len(result['paragraphs'])}")
        return result


# Default instance (used by scraper.py and others)
default_preprocessor = ParagraphPreprocessor()